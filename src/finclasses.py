import random
import pandas as pd
from connectors import PGConn

from debug import log_message


class Ticker:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data

        self.preprocess_data()

    def preprocess_data(self):
        self.data = self.data.set_index('date', drop=True).sort_index()
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.data['row_id'] = list(range(1, len(self.data) + 1))
        self.data['close'] = self.data['close'].astype(float)

    def calculate_moving_average(self, window):
        self.data[f'moving_average_{window}'] = self.data['close'].rolling(
            window=window).mean()

    def calculate_rsi(self, window):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def calculate_macd(self, short_window, long_window, signal_window):
        short_ema = self.data['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data['close'].ewm(span=long_window, adjust=False).mean()
        self.data['macd'] = short_ema - long_ema
        self.data['signal'] = self.data['macd'].ewm(span=signal_window, adjust=False).mean()

    def calculate_bollinger_bands(self, window):
        self.data['moving_average'] = self.data['close'].rolling(window=window).mean()
        self.data['std_dev'] = self.data['close'].rolling(window=window).std()
        self.data['upper_band'] = self.data['moving_average'] + (self.data['std_dev'] * 2)
        self.data['lower_band'] = self.data['moving_average'] - (self.data['std_dev'] * 2)

    def calculate_ema(self, window):
        self.data['ema'] = self.data['close'].ewm(span=window, adjust=False).mean()


class Finalysis:
    """
    The Finalysis class features as a super class that different methods of financial analysis can be inherited from.
    The class is designed to allow for simple integration with the glint dashboard by providing a common interface for
    handling ticker data and streaming it to the dashboard.
    The class is initialised with an instance of the Ticker class, which is a simple data structure that holds the
    historical data of a stock which is pulled from the Glint postgres database.
    """
    def __init__(self, ticker):
        self.ticker = ticker

    def find_return_points(self, window_size, return_threshold):
        """
        This method finds all points in the data where investing in the stock would have resulted in a return of
        greater than the return_threshold in the window_size. N.B. If flexible_window is False, the window is fixed,
        so the return is only considered at the end of the window. If flexible_window is True, the return is considered
        at any point in the window provided it crosses the return_threshold.
        :param window_size (int): The number of data points to look across.
        :param return_threshold (float): The minimum return, as a fraction (e.g. for 5% return = 0.05)
        :param flexible_window (bool): Whether the return can be at any point in the window, or only at the end.
        required to be considered a return point.
        """
        df = self.ticker.data
        return_vals = df['close'].pct_change()

        return_pts = pd.concat([return_vals.rolling(window=win_size).sum().shift(-1*win_size) > return_threshold for
                                win_size in range(1, window_size)], axis=1).any(axis=1)
        self.ticker.data['return_points'] = return_pts

        return return_pts

    def apply_strategy(self, starting_cash=1000, stop_loss=0.02, take_profit=0.05):
        holding = False
        buy_price = 0
        self.ticker.data['signal'] = False
        portfolio = []
        for idx, row in self.ticker.data.iterrows():
            # Signal method (random for testing)
            if self.signal(idx) and not holding:
                self.ticker.data.at[idx, 'signal'] = True
                buy_price = row['close']
                holding = True

            if holding:
                # Determine if stop loss or take profit has been hit
                if row['close'] < buy_price * (1 - stop_loss) or row['close'] > buy_price * (1 + take_profit):
                    starting_cash = starting_cash * (row['close'] / buy_price)
                    holding = False

            portfolio.append(starting_cash)

        self.ticker.data['portfolio'] = portfolio

    def signal(self, idx):
        """
        This method is designed to be overridden by subclasses to build a specific strategy to the stock data.
        When not overridden, it simply applies a random strategy to the data for testing purposes.
        """
        log_message('WARNING: apply_strategy method has not been overridden, applying random strategy for testing purposes.')

        if random.random() > 0.5:
            return True
        else:
            return False


class TestAnalytics(Finalysis):
    """
    The MovingAverage class is a subclass of the Finalysis class, and is designed to provide a simple interface for
    calculating moving averages on stock data.
    """
    def __init__(self, ticker):
        super().__init__(ticker)
        self.ticker.calculate_moving_average(20)
        self.ticker.calculate_moving_average(50)
        self.ticker.calculate_moving_average(100)
        self.ticker.calculate_bollinger_bands(20)
        self.ticker.calculate_rsi(14)

    def signal(self, idx):
        """
        This method is designed to be overridden by subclasses to build a specific strategy to the stock data.
        When not overridden, it simply applies a random strategy to the data for testing purposes.
        """
        data = self.ticker.data

        if data['rsi'][idx] < 30 and data['close'][idx] < data['lower_band'][idx]:
            return True
        else:
            return False


def main():
    conn = PGConn()
    tickers = conn.get_tickers()
    ticker_data = conn.get_ticker_data('AAPL')
    ticker = Ticker(tickers[0], ticker_data)

    test = TestAnalytics(ticker)
    test.find_return_points(window_size=20, return_threshold=0.05)
    test.apply_strategy()


if __name__ == '__main__':
    main()