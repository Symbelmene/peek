import pandas as pd
from connectors import PGConn

from finclasses import Portfolio
from strategies import ticker_should_be_sold, down_select_tickers


def output_summary(data, date, pf):
    holdings = pf.calculate_value_of_holdings(data)
    print(f'Portfolio value on {date}: Cash: {round(pf.cash, 2)}, '
          f'Holdings: {round(holdings, 2)}, '
          f'Total: {round(pf.cash + holdings, 2)}')


def buy_stocks(BUY_AMOUNT, data, pf):
    down_selected_tickers = down_select_tickers(data)
    for ticker in down_selected_tickers:
        if ticker not in [e[0] for e in pf.portfolio] and pf.cash >= BUY_AMOUNT and len(pf.portfolio) < 5:
            curr_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
            pf.buy(ticker, BUY_AMOUNT / curr_ticker_value, BUY_AMOUNT)


def sell_stocks(data, pf):
    for ticker, amount, value in pf.portfolio:
        if ticker_should_be_sold(ticker, amount, value, data):
            curr_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
            pf.sell(ticker, curr_ticker_value)


def main():
    START_DATE = '2022-05-01'
    END_DATE = '2022-08-30'
    START_AMOUNT = 10000
    BUY_AMOUNT = 500

    pg_conn = PGConn()

    pf = Portfolio(START_AMOUNT)

    # Iterate through days from start to end date
    date = pd.to_datetime(START_DATE)
    while date <= pd.to_datetime(END_DATE):
        data = pg_conn.fetch_period_data(date, days_history=30)

        sell_stocks(data, pf)

        buy_stocks(BUY_AMOUNT, data, pf)

        output_summary(data, date, pf)

        date += pd.Timedelta(days=1)


if __name__ == '__main__':
    main()