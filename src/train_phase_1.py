import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical

from model import build_lstm_model
from config import Config
cfg = Config()


class Inline_Generator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, batch_size, history_size):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.history_size = history_size

    def __len__(self):
        return (np.ceil(len(self.file_paths) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch = np.array([np.load(x) for x in batch])
        x, y = batch[:, :self.history_size, :], batch[:, self.history_size:, -1]
        return x, y


def standardise_data(arr):
    # Standardises columns in a 2D array using np.apply_along_axis and a lambda function
    return np.apply_along_axis(lambda x: (x - x.min()) / (x.max() - x.min()), 0, arr)


def get_label(file_path):
    return int(file_path.split('/')[-1].split('_')[1][0])


def prepare_phase_1_train_set(data_dir, batch_size, history_size):
    ticker_dirs = [f'{data_dir}/{dir}' for dir in os.listdir(data_dir)]

    file_details_list = []
    for ticker_dir in ticker_dirs:
        ticker_files = [{'file' : f'{ticker_dir}/{file}', 'ticker' : ticker_dir.split('/')[-1]} for file in os.listdir(ticker_dir)]
        file_details_list += ticker_files

    df = pd.DataFrame(file_details_list)
    df['file_name'] = df['file'].apply(lambda x: x.split('/')[-1])

    # Down select tickers by data size
    ticker_groups = df.groupby('ticker')
    ticker_group_lengths = ticker_groups.apply(len)
    median_group_length = ticker_group_lengths.median()

    # Only select tickers within 90% of the median group length
    selected_tickers = ticker_group_lengths[ticker_group_lengths > 0.8 * median_group_length].index

    df_short = df[df['ticker'].isin(selected_tickers)]

    file_list = df_short['file'].values
    train_files = file_list[:int(len(file_list) * 0.8)]
    test_files = file_list[int(len(file_list) * 0.8):]

    random.shuffle(train_files)
    random.shuffle(test_files)

    return (Inline_Generator(train_files, batch_size, history_size),
            Inline_Generator(test_files, batch_size, history_size))


def train_phase_1(batch_size, num_epochs, history_size, data_dir, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create dataset
    train_batch_generator, test_batch_generator = prepare_phase_1_train_set(data_dir, batch_size, history_size)

    # Build model
    sample_data = np.load(train_batch_generator.file_paths[0])
    model = build_lstm_model(historySize=sample_data.shape[0], numHistoryVars=sample_data.shape[1])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=1)

    # Train model
    model.fit(x=train_batch_generator,
              epochs=num_epochs,
              batch_size=batch_size,
              validation_data=test_batch_generator,
              callbacks=[tensorboard_callback,
                         tf.keras.callbacks.ModelCheckpoint(f'{model_dir}/stock_model_phase_1.keras',
                                                            monitor='val_loss',
                                                            save_best_only=True)])

    model.save(f'{model_dir}/stock_model_phase_1.keras')


def main():
    data_dir = '/home/christ/findata/data'
    model_dir = '../models'
    batch_size = 256
    num_epochs = 50
    history_size = 256

    # Check GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)

    train_phase_1(batch_size, num_epochs, history_size, data_dir, model_dir)

    #test_phase_1(data_dir, model_dir)


if __name__ == '__main__':
    main()
