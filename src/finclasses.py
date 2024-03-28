

class Portfolio:
    def __init__(self, initial_investment):
        self.initial_investment = initial_investment
        self.portfolio = []
        self.cash = initial_investment

    def buy(self, ticker, num_stocks, value):
        if self.cash >= value:
            self.portfolio.append((ticker, num_stocks, value))
            self.cash -= value

    def sell(self, ticker, current_value):
        indexes_to_remove = []
        for i, (t, num, val) in enumerate(self.portfolio):
            if t == ticker:
                self.cash += current_value * num
                indexes_to_remove.append(i)

        for i in indexes_to_remove:
            del self.portfolio[i]

    def calculate_value_of_holdings(self, data):
        value = 0
        for ticker, amount, _ in self.portfolio:
            current_ticker_value = data[data['ticker'] == ticker].iloc[-1]['adj_close']
            value += current_ticker_value * amount

        return value

