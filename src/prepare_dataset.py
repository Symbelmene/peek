"""
Find stocks which are good for trading on mean reversion
"""
import os
import numpy as np
import pandas as pd
from connectors import PGConn


def get_return_points_above_percentage(ticker_data, return_percentage, return_window):
    perc_change = ticker_data['adj_close'] / ticker_data['adj_close'].shift(1) - 1

    if return_window > 1:
        rolling_rets = pd.concat([perc_change.rolling(window=i).sum() for i in range(1, return_window)], axis=1)
    else:
        rolling_rets = pd.DataFrame(perc_change)

    ret_masks_all = rolling_rets.applymap(lambda x: x > return_percentage)
    ret_mask_any = ret_masks_all.apply(lambda x: any(x), axis=1)
    return ret_mask_any


def extract_dataset_from_ticker(data, mask, input_length=30, return_futures=False, future_length=5):
    locs = [(idx - input_length, idx) for idx, _ in enumerate(list(mask)) if idx - input_length >= 0]
    labels = mask.to_numpy()
    train_set = np.array([data[start:end] for start, end in locs])
    labels = labels[input_length:]
    if not return_futures:
        return train_set, labels

    futures = np.array([labels[idx:idx+future_length] for idx in range(len(labels)-7)])
    return train_set, labels, futures


def main():
    MIN_RETURN = 0.03
    RETURN_WINDOW = 5
    MIN_RETURN_PERC_FOR_TICKER_TO_BE_CONSIDERED = 0.05

    pg_conn = PGConn()
    tickers = pg_conn.get_tickers()

    print(f'Ticker: Return points above {MIN_RETURN * 100}%')
    train_set, label_set = [], []
    for ticker in tickers:
        data = pg_conn.fetch_ticker_data(ticker)
        return_points = get_return_points_above_percentage(data, MIN_RETURN, RETURN_WINDOW)
        percentage_return_points = return_points.sum() / len(data)
        print(f'{ticker}: {round(100*percentage_return_points, 1)}% points')
        if percentage_return_points > MIN_RETURN_PERC_FOR_TICKER_TO_BE_CONSIDERED:
            ticker_train_data, ticker_train_labels = extract_dataset_from_ticker(data, return_points)
            train_set.append(ticker_train_data)
            label_set.append(ticker_train_labels)

    X = np.vstack(train_set)
    Y = np.hstack(label_set)
    print(X.shape, Y.shape)

    if not os.path.exists('../data'):
        os.makedirs('../data')

    np.save('../data/X.npy', X)
    np.save('../data/Y.npy', Y)


if __name__ == '__main__':
    main()