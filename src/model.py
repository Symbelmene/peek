import tensorflow as tf
from keras import layers, losses


def build_lstm_model(history_size, future_size, num_history_vars):
    # Builds and Functional LSTM model using the windowSize and numStocks as input and numActions as output
    input_layer = layers.Input(shape=(num_history_vars, history_size))

    x = layers.LSTM(32, return_sequences=True)(input_layer)
    x = layers.LSTM(16, return_sequences=True)(x)
    x = layers.LSTM(8, return_sequences=False)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(18, activation='sigmoid')(x)
    x = layers.Dense(9, activation='sigmoid')(x)
    x = layers.Dense(9, activation='sigmoid')(x)

    outputLayer = layers.Dense(future_size, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=outputLayer)

    model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['accuracy'])

    print(model.summary())

    return model

