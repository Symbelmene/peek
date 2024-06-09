import wandb
import numpy as np
import tensorflow as tf
from keras import layers, losses
from keras.utils import to_categorical

from config import Config
cfg = Config()

wandb.login(key=cfg.WANDB_API_KEY)


def buildLSTMModel(historySize, numHistoryVars):
    # Builds and Functional LSTM model using the windowSize and numStocks as input and numActions
    # as output
    inputLayer = layers.Input(shape=(historySize, numHistoryVars))
    x = layers.LSTM(numHistoryVars, return_sequences=True)(inputLayer)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='sigmoid')(x)
    x = layers.Dense(16, activation='sigmoid')(x)
    x = layers.Dense(8, activation='sigmoid')(x)
    x = layers.Dense(4, activation='sigmoid')(x)
    outputLayer = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputLayer, outputs=outputLayer)
    model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=['accuracy'])
    return model


def trim_dataset_to_matching_labels(X, Y):
    # Trims the dataset so that equal number of 0 and 1 labels are present
    # This is done to prevent the model from learning to predict only one class
    X_0 = X[Y == 0]
    X_1 = X[Y == 1]
    min_length = min(X_0.shape[0], X_1.shape[0])
    X_0 = X_0[:min_length]
    X_1 = X_1[:min_length]
    X = np.vstack([X_0, X_1])
    Y = np.hstack([np.zeros(min_length), np.ones(min_length)])

    print(f'Reduced dataset to {X.shape[0]} samples with {min_length} samples of each class')
    return X, Y


def standardise_data(arr):
    # Standardises columns in a 2D array using np.apply_along_axis and a lambda function
    return np.apply_along_axis(lambda x: (x - x.min()) / (x.max() - x.min()), 0, arr)


def prepareTrainingDataset():
    X = np.load('../../glint_dashboard/local_data/norm_data.npy')
    Y = np.load('../../glint_dashboard/local_data/norm_labels.npy')

    # One-hot encode the labels
    Y = np.array([to_categorical(y, num_classes=2) for y in Y])

    # Normalise data on a per-item basis
    X = np.apply_along_axis(standardise_data, 0, X)

    keys = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    # Split the data into training and validation sets
    split = int(X.shape[0] * 0.8)
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]

    return x_train, y_train, x_val, y_val


def main():
    RUN_CONFIG = {
        'EPOCHS': 10,
        'BATCH_SIZE': 256,
        'HISTORY_SIZE': 30,
        'NUM_HISTORY_VARS': 5
    }

    # Create dataset
    x_train, y_train, x_val, y_val = prepareTrainingDataset()

    run = wandb.init(
        project="peek",
        config={
        })

    # Build model
    model = buildLSTMModel(historySize=x_train.shape[1],
                           numHistoryVars=x_train.shape[2])

    print(model.summary())

    # Train model
    model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_val, y_val))


if __name__ == '__main__':
    main()