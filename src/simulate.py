import pandas as pd
from connectors import PGConn

from strategy import sell_on_profit_or_stop_loss
from finclasses import Portfolio


def output_summary(data, date, pf):
    holdings = pf.calculate_value_of_holdings(data)
    print(f'Portfolio value on {date}: Cash: {round(pf.cash, 2)}, '
          f'Holdings: {round(holdings, 2)}, '
          f'Total: {round(pf.cash + holdings, 2)}')


def buy_stocks(RUN_CFG, data, pf, strat):
    ticker_groups = data.groupby('ticker')
    pred_groups = [(ticker, t_data[:RUN_CFG.HISTORY_SIZE]) for ticker, t_data in ticker_groups if
                   len(t_data[:RUN_CFG.HISTORY_SIZE]) == RUN_CFG.HISTORY_SIZE]

    signals = strat.infer_signal(pred_groups)

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


def backtest(RUN_CFG, strat):
    pg_conn = PGConn()

    pf = Portfolio(RUN_CFG.START_AMOUNT)

    # Iterate through days from start to end date
    date = pd.to_datetime(RUN_CFG.START_DATE)
    while date <= pd.to_datetime(RUN_CFG.END_DATE):
        data = pg_conn.fetch_period_data(date, days_history=RUN_CFG.HISTORY_SIZE)

        sell_stocks(data, pf)

        buy_stocks(RUN_CFG, data, pf, strat)

        output_summary(data, date, pf)

        date += pd.Timedelta(days=1)
