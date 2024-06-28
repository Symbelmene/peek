import os
import shutil
import numpy as np
from tqdm import tqdm
from finclasses import Ticker, Finalysis
from connectors import PGConn


def remove_duplicates():
    pg_conn = PGConn()
    pg_conn.find_and_remove_duplicate_entries()


def fetch_ticker_data(ticker):
    pg_conn = PGConn()
    return Ticker(ticker, pg_conn.get_ticker_data(ticker))


def get_num_return_points(ticker_obj, return_threshold, window_size):
    ticker_analytics = Finalysis(ticker_obj)

    if ticker_obj.ticker == 'AAPL':
        x = 1

    ticker_analytics.find_return_points(window_size, return_threshold)


    tdf = ticker_analytics.ticker.data
    num_return_points = len(tdf[tdf['return_points']])
    return {'valid': num_return_points, 'total': len(tdf)}


def create_dataset(min_return_point_pct, return_threshold, train_history, train_future, train_columns, window_size, data_dir):
    # Iterate tickers and get number of return points
    pg_conn = PGConn()
    tickers = pg_conn.get_tickers(sector='Information Technology')

    for ticker in tqdm(tickers):
        ticker_obj = fetch_ticker_data(ticker)
        ret_pts_dict = get_num_return_points(ticker_obj, return_threshold, window_size)
        if ret_pts_dict['total'] > 0 and (ret_pts_dict['valid'] / ret_pts_dict['total']) > min_return_point_pct:
            ticker_dataset = process_ticker_to_dataset(ticker_obj, train_history, train_future, train_columns, data_dir)
            if ticker_dataset is not None:
                expected_shape = (train_history + train_future, len(train_columns))
                ticker_dataset = [data for data in ticker_dataset if data.shape == expected_shape]
                write_data_to_file(ticker, ticker_dataset, data_dir)


def add_indicators(ticker):
    ticker.calculate_moving_average(20)
    ticker.calculate_moving_average(50)
    ticker.calculate_bollinger_bands(20)
    ticker.calculate_rsi(14)
    ticker.calculate_macd(12, 26, 9)
    ticker.calculate_ema(20)
    return ticker


def process_ticker_to_dataset(ticker, train_history, train_future, train_columns, data_dir):
    try:
        train_data = ticker.data[train_columns][100:]

        data = [train_data.iloc[i - train_history:i + train_future].values.astype(float)
                for i in range(train_history, len(train_data - train_future))]

        # Normalise the data and convert labels to 0/1
        norm_data = [np.array((x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))) for x in data]
    except:
        print(f'Error processing ticker {ticker.ticker}')
        return None
    return norm_data


def write_data_to_file(ticker, ticker_data, data_dir):
    out_dir = f'{data_dir}/{ticker}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = 0
    for data in ticker_data:
        if data is None:
            print(f'Error writing data for {ticker} {idx}')
            continue
        if np.isnan(data).any():
            print(f'Error writing data for {ticker} {idx}')
            continue
        np.save(f'{out_dir}/{idx}.npy', data)
        idx += 1


def main():
    window_size = 3
    return_threshold = 0.03
    min_return_point_pct = 0.2

    train_history = 256
    train_future = 3

    data_dir = '/home/christ/findata/data'

    train_columns = ['high', 'low', 'open', 'adj_close']

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    os.makedirs(data_dir)

    #              min_return_point_pct, return_threshold, train_history, train_future, train_columns, window_size, data_dir
    create_dataset(min_return_point_pct, return_threshold, train_history, train_future, train_columns, window_size, data_dir)


if __name__ == '__main__':
    main()
