"""
Find stocks which are good for trading on mean reversion
"""
import os
import numpy as np
import pandas as pd
from connectors import PGConn


def get_return_points_above_percentage(ticker_data, return_percentage, return_window):
    perc_change = ticker_data['adj_close'] / ticker_data['adj_close'].shift(1) - 1
    rolling_rets = pd.concat([perc_change.rolling(window=i).sum() for i in range(1, return_window)], axis=1)
    ret_masks_all = rolling_rets.applymap(lambda x: x > return_percentage)
    ret_mask_any = ret_masks_all.apply(lambda x: any(x), axis=1)
    return ret_mask_any


def extract_dataset_from_ticker(data, mask, input_length=30):
    locs = [(idx - input_length, idx) for idx, _ in enumerate(list(mask)) if mask[idx] and idx - input_length >= 0]
    arr = data.to_numpy()

    dataset = np.array([arr[start:end] for start, end in locs])
    return dataset


def main():
    MIN_RETURN = 0.05
    RETURN_WINDOW = 7
    MIN_RETURN_PERC_FOR_TICKER_TO_BE_CONSIDERED = 0.05

    pg_conn = PGConn()
    tickers = pg_conn.get_tickers()

    print(f'Ticker: Return points above {MIN_RETURN * 100}%')
    tickers_set = []
    for ticker in tickers:
        data = pg_conn.fetch_ticker_data(ticker)
        return_points = get_return_points_above_percentage(data, MIN_RETURN, RETURN_WINDOW)
        percentage_return_points = return_points.sum() / len(data)
        print(f'{ticker}: {round(100*percentage_return_points, 1)}% days')
        if percentage_return_points > MIN_RETURN_PERC_FOR_TICKER_TO_BE_CONSIDERED:
            ticker_train_data = extract_dataset_from_ticker(data, return_points)
            tickers_set.append(ticker_train_data)

    data_set = np.vstack(tickers_set)
    print(data_set.shape)

    if not os.path.exists('../data'):
        os.makedirs('../data')

    np.save('../data/data_set.npy', data_set)


if __name__ == '__main__':
    main()