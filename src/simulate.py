import pandas as pd
from connectors import PGConn

from strategy import sell_on_profit_or_stop_loss, Finalytics
from finclasses import Portfolio


def output_summary(data, date, pf):
    holdings = pf.calculate_value_of_holdings(data)
    print(f'Portfolio value on {date}: Cash: {round(pf.cash, 2)}, '
          f'Holdings: {round(holdings, 2)}, '
          f'Total: {round(pf.cash + holdings, 2)}')


def buy_stocks(RUN_CFG, data, pf, strategy):
    signals = strategy.infer_signal(data)

    # Select top 5 signals
    signals = sorted(signals, key=lambda x: x[1], reverse=True)[:5]

    for ticker, signal in signals:
        if signal > 0.5:
            if ticker not in [e[0] for e in pf.portfolio] and pf.cash >= RUN_CFG.BUY_AMOUNT and len(pf.portfolio) < 5:
                curr_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
                pf.buy(ticker, RUN_CFG.BUY_AMOUNT / curr_ticker_value, RUN_CFG.BUY_AMOUNT)


def sell_stocks(data, pf):
    for ticker, amount, value in pf.portfolio:
        if sell_on_profit_or_stop_loss(ticker, amount, value, data, take_profit=0.05, stop_loss=0.02):
            curr_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
            pf.sell(ticker, curr_ticker_value)


def backtest(RUN_CFG, strategy):
    pg_conn = PGConn()

    pf = Portfolio(RUN_CFG.START_AMOUNT)

    # Iterate through days from start to end date
    date = pd.to_datetime(RUN_CFG.START_DATE)
    while date <= pd.to_datetime(RUN_CFG.END_DATE):
        data = pg_conn.fetch_period_data(date, days_history=RUN_CFG.HISTORY_SIZE)

        sell_stocks(data, pf)

        buy_stocks(RUN_CFG, data, pf, strategy)

        output_summary(data, date, pf)

        date += pd.Timedelta(days=1)


class RUN_CONFIG:
    def __init__(self):
        # Backtesting configuration
        self.START_DATE = '2023-01-01'
        self.END_DATE = '2024-01-01'
        self.START_AMOUNT = 10000
        self.BUY_AMOUNT = 500
        self.HISTORY_SIZE = 80


def main():
    cfg = RUN_CONFIG()
    backtest(cfg, Finalytics())


if __name__ == '__main__':
    main()