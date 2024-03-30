import model
import numpy as np


class PeekModel:
    def __init__(self, model):
        self._model = model

    def infer_signal(self, ticker_data):
        keys = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        x = np.array([model.standardise_data(d[1][keys].values) for d in ticker_data])
        preds = self._model.predict(x)
        tickers = [t[0] for t in ticker_data]
        buy_conf = [pred[1] for pred in preds]
        return list(zip(tickers, buy_conf))


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