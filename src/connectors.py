import pandas as pd
import psycopg2 as pg
from datetime import timedelta
from config import Config
cfg = Config()


class PGConn:
    def __init__(self):
        # Switch to the findata database
        self.conn = pg.connect(dbname=cfg.STORER_DB_NAME,
                               user=cfg.STORER_USER, password=cfg.STORER_PASSWORD,
                               host=cfg.STORER_HOST, port=cfg.STORER_PORT)

    def get_tickers(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT ticker FROM stock_data;")
            tickers = cursor.fetchall()
        return [t[0] for t in tickers]

    def fetch_all(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM stock_data;")
            data = cursor.fetchall()

        keys = ['index', 'date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = pd.DataFrame(data, columns=keys)
        df = df.drop(columns=['index']).set_index('date')

        for key in ['open', 'high', 'low', 'close', 'adj_close']:
            df[key] = df[key].apply(float)

        return df

    def fetch_period_data(self, date, days_history):
        date_to = pd.to_datetime(date)
        date_from = date_to - timedelta(days=days_history)

        # Fetch data from the database
        with self.conn.cursor() as cursor:
            query = "SELECT * FROM stock_data WHERE date BETWEEN %s AND %s;"
            cursor.execute(query, (date_from, date_to))
            data = cursor.fetchall()

        keys = ['index', 'date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = pd.DataFrame(data, columns=keys)
        df = df.drop(columns=['index']).set_index('date')

        for key in ['open', 'high', 'low', 'close', 'adj_close']:
            df[key] = df[key].apply(float)

        return df

    def fetch_ticker_data(self, ticker):
        query = f"SELECT * FROM stock_data WHERE ticker = %s;"
        with self.conn.cursor() as cursor:
            cursor.execute(query, (ticker,))
            data = cursor.fetchall()

        keys = ['index', 'date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = pd.DataFrame(data, columns=keys)
        df = df.drop(columns=['index', 'ticker']).set_index('date')

        for key in ['open', 'high', 'low', 'close', 'adj_close']:
            df[key] = df[key].apply(float)

        return df