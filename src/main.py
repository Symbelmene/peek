
"""
This code will -
From a given START_DATE and END_DATE, run a simulation of stock prices for a given number of days.
Data will be fetched from the glint database on an hourly basis
"""
import time
import pandas as pd
from connectors import PGConn


class Portfolio:
    def __init__(self, initial_investment):
        self.initial_investment = initial_investment
        self.portfolio = []
        self.cash = initial_investment

    def buy(self, ticker, num_stocks, value):
        if self.cash >= value:
            self.portfolio.append((ticker, num_stocks, value))
            self.cash -= value

    def sell(self, ticker, current_value):
        indexes_to_remove = []
        for i, (t, num, val) in enumerate(self.portfolio):
            if t == ticker:
                self.cash += current_value * num
                indexes_to_remove.append(i)

        for i in indexes_to_remove:
            del self.portfolio[i]

    def calculate_value_of_holdings(self, data):
        value = 0
        for ticker, amount, _ in self.portfolio:
            current_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
            value += current_ticker_value * amount

        return value


def down_select_tickers(data):
    good_tickers = []

    ticker_groups = data.groupby('ticker')
    for ticker, group in ticker_groups:
        data_norm = 1 - group['adj_close'] / group['adj_close'].shift(1)
        volatility = data_norm.std()
        moving_av_30pt = group['adj_close'].rolling(window=30).mean()
        dev_30 = group['adj_close'] - moving_av_30pt

        if volatility > 0.05 and dev_30.iloc[-1] < 0:
            good_tickers.append(ticker)

    return good_tickers


def ticker_should_be_sold(ticker, amount, value, data):
    current_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
    current_holdings_value = amount * current_ticker_value

    if 1 - (current_holdings_value / value) > 0.1:
        return True
    return False


def main():
    START_DATE = '2022-05-01'
    END_DATE = '2022-08-30'
    INVESTMENT = 10000
    BUY_AMOUNT = 500

    pg_conn = PGConn()

    pf = Portfolio(INVESTMENT)

    # Iterate through days from start to end date
    date = pd.to_datetime(START_DATE)
    while date <= pd.to_datetime(END_DATE):
        date_str = date.strftime('%Y-%m-%d')

        # Fetch data from the database
        data = pg_conn.fetch_data(date_str, days_history=30)

        # Check if any of current portfolio stocks need to be sold
        for ticker, amount, value in pf.portfolio:
            if ticker_should_be_sold(ticker, amount, value, data):
                curr_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
                pf.sell(ticker, curr_ticker_value)

        # Process the data
        down_selected_tickers = down_select_tickers(data)

        # Decide which stocks to buy
        for ticker in down_selected_tickers:
            if ticker not in [e[0] for e in pf.portfolio] and pf.cash >= BUY_AMOUNT and len(pf.portfolio) < 5:
                curr_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
                pf.buy(ticker, BUY_AMOUNT / curr_ticker_value, BUY_AMOUNT)

        # Summarise the portfolio
        holdings = pf.calculate_value_of_holdings(data)
        print(f'Portfolio value on {date_str}: Cash: {round(pf.cash, 2)}, '
              f'Holdings: {round(holdings, 2)}, '
              f'Total: {round(pf.cash + holdings, 2)}')

        date += pd.Timedelta(days=1)


if __name__ == '__main__':
    main()