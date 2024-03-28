



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
