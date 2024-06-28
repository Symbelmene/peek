import traceback
import pandas as pd
import psycopg2 as pg

from debug import log_message
from config import Config
cfg = Config()


class PGConn:
    def __init__(self):
        self.conn = pg.connect(dbname=cfg.STORER_DB_NAME,
                               user=cfg.STORER_USER, password=cfg.STORER_PASSWORD,
                               host=cfg.STORER_HOST, port=cfg.STORER_PORT)

    def get_tickers_for_sector(self, sector_name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT t.id, t.ticker FROM tickers t JOIN sectors s ON t.sector_id = s.id WHERE s.sector = %s",
                           (sector_name,))
            return [row[1] for row in cursor.fetchall()]

    def get_sectors(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT sector FROM sectors")
            return [row[0] for row in cursor.fetchall()]

    def get_tickers(self, sector=None):
        if sector is None:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT ticker FROM tickers")
                return [row[0] for row in cursor.fetchall()]
        else:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT t.ticker FROM tickers t JOIN sectors s ON t.sector_id = s.id WHERE s.sector = %s",
                               (sector,))
                return [row[0] for row in cursor.fetchall()]

    def get_ticker_data(self, ticker, as_dataframe=True):
        query = """SELECT * FROM stock_data_day sdd INNER JOIN tickers t ON
                   sdd.ticker_id = t.id WHERE ticker = %s"""
        try:
            cur = self.conn.cursor()
            cur.execute(query, (ticker,))
            ticker_data = cur.fetchall()
        except Exception as e:
            log_message(f'Error getting all feature details: {e}')
            log_message(traceback.format_exc())
            self.conn.rollback()
            return False

        if as_dataframe:
            headers = [desc.name for desc in cur.description]
            ticker_data = pd.DataFrame(ticker_data, columns=headers)

        return ticker_data

    def find_and_remove_duplicate_entries(self):
        tickers = self.get_tickers()
        for ticker in tickers:
            remove_list = []
            td = self.get_ticker_data(ticker)
            for name, group in td.groupby('date'):
                if len(group) > 1:
                    remove_list += list(group['id'][1:])
            if remove_list:
                print(f'{ticker}: {len(remove_list)} duplicates')
                self.delete_entry_by_row_ids(remove_list)

    def delete_entry_by_row_ids(self, row_id_list):
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM stock_data_day WHERE id IN %s", (tuple(row_id_list),))
            self.conn.commit()
