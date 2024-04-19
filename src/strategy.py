import numpy as np
import pandas as pd
from tqdm import tqdm

class PeekModel:
    def __init__(self, model):
        self._model = model

    def infer_signal(self, ticker_data):
        keys = ['open', 'high', 'low', 'adj_close', 'volume']
        x = np.array([model.standardise_data(d[1][keys].values) for d in ticker_data])
        preds = self._model.predict(x)
        tickers = [t[0] for t in ticker_data]
        buy_conf = [pred[1] for pred in preds]
        return list(zip(tickers, buy_conf))


class Finalytics:
    def __init__(self):
        pass

    def infer_signal(self, ticker_data):
        groups = ticker_data.groupby('ticker')
        signal_list = []
        for name, group in tqdm(groups, total=len(groups)):
            try:
                signal = self.calculate_predictions(group)
                signal_list.append((name, signal))
            except:
                print(f'Error in ticker {name}')

        return signal_list

    def calculate_predictions(self, group):
        ma30 = group['adj_close'].rolling(window=30).mean()
        ma50 = group['adj_close'].rolling(window=50).mean()
        std = group['adj_close'].rolling(window=30).std()
        boll_band_low = ma30 - 2 * std
        boll_band_up = ma30 + 2 * std
        rsi = calculate_rsi(group['adj_close'])
        metrics = pd.DataFrame({
            'ma30': ma30,
            'ma50': ma50,
            'boll_band_low': boll_band_low,
            'boll_band_up': boll_band_up,
            'rsi': rsi
        })
        # Check if stock is above bollinger band and rsi is below 30
        metrics['buy'] = (group['adj_close'] > boll_band_up) & (rsi < 30)
        metrics = metrics.dropna()
        # Check if stock is below bollinger band and rsi is above 70
        metrics['sell'] = (group['adj_close'] < boll_band_low) & (rsi > 70)
        metrics = metrics.dropna()
        # Infer buy/sell signals
        last_row = metrics.iloc[-1]
        signal = 1 if last_row['buy'] else -1 if last_row['sell'] else 0
        return signal


def sell_on_profit_or_stop_loss(ticker, amount, value, data, take_profit, stop_loss):
    current_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
    current_holdings_value = amount * current_ticker_value

    if 1 - (current_holdings_value / value) > take_profit:
        # Take Profit
        return True
    if 1 - (current_holdings_value / value) < -stop_loss:
        # Stop Loss
        return True
    # Hold
    return False


def calculate_rsi(close_values):
    df = pd.DataFrame(close_values)
    df['change'] = df['adj_close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)

    df['avg_gain'] = rma(df.gain.to_numpy(), 14)
    df['avg_loss'] = rma(df.loss.to_numpy(), 14)

    df['rs'] = df.avg_gain / df.avg_loss
    df['rsi'] = 100 - (100 / (1 + df.rs))
    return df['rsi']


def rma(x, n):
    """Running moving average"""
    a = np.full_like(x, np.nan)
    a[n] = x[1:n + 1].mean()
    for i in range(n + 1, len(x)):
        a[i] = (a[i - 1] * (n - 1) + x[i]) / n
    return a
