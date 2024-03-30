import model
import dataset
import strategy
import simulate


class RUN_CONFIG:
    def __init__(self):
        # Train data configuration
        self.MIN_RETURN = 0.03
        self.RETURN_WINDOW = 5
        self.MIN_RETURN_PERC_FOR_TICKER_TO_BE_CONSIDERED = 0.05

        # Model training configuration
        self.EPOCHS = 20
        self.BATCH_SIZE = 256
        self.HISTORY_SIZE = 30
        self.NUM_HISTORY_VARS = 5

        # Backtesting configuration
        self.START_DATE = '2023-01-01'
        self.END_DATE = '2024-01-01'
        self.START_AMOUNT = 10000
        self.BUY_AMOUNT = 500


def main():
    RUN_CFG = RUN_CONFIG()

    # Prepare the dataset
    # dataset.create(RUN_CFG)

    # Train the model
    # trained_model = model.train_model(RUN_CFG)

    # Backtest the model
    current_model_path = '../models/magic-dream-7-model.h5'
    current_model = model.load_model(current_model_path)

    strat = strategy.PeekModel(current_model)

    simulate.backtest(RUN_CFG, strat)


if __name__ == '__main__':
    main()