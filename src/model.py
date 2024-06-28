import tensorflow as tf
from keras import layers, losses


def build_lstm_model(historySize, numHistoryVars):
    # Builds and Functional LSTM model using the windowSize and numStocks as input and numActions as output
    input_layer = layers.Input(shape=(historySize, numHistoryVars))
    x = layers.Conv1D(64, 3, activation='relu')(input_layer)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(32, 3, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(16, 3, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(18, activation='sigmoid')(x)
    x = layers.Dense(9, activation='sigmoid')(x)
    x = layers.Dense(9, activation='sigmoid')(x)
    outputLayer = layers.Dense(3, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=outputLayer)
    model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['accuracy'])
    print(model.summary())
    return model

