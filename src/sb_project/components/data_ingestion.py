from dataclasses import dataclass
from pathlib import Path
import os
from sb_project.utils.common import read_yaml, create_directories

import yfinance as yf
from curl_cffi import requests
import time
import pandas as pd


CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    tickers : list[str]
    start_date : str
    interval : str
    auto_adjust : bool

# create attribute: config file path
# create method: get the the configs from the file
class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        yf_config = self.config.yfinance_config
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            tickers=yf_config.tickers,
            start_date=yf_config.start_date,
            interval=yf_config.interval,
            auto_adjust=yf_config.auto_adjust
        )
        return data_ingestion_config
    
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.ticker_df = None


    def download_data(self, ticker):
        """
        Download stock data from Yahoo Finance.
        """
        session = requests.Session(impersonate="chrome")
        start_date = pd.to_datetime(self.config.start_date)
        interval = self.config.interval
        data = yf.download(ticker, start=start_date, interval=interval, auto_adjust=False, session=session)
        data.columns = data.columns.droplevel(1)  # Drop the first level of the column index
        # data.reset_index(inplace=True)  # Reset the index to make 'Date' a column
        return data

    def get_ticker_data(self):
        stocks_df = pd.DataFrame()
        for i,ticker in enumerate(self.config.tickers):
            print(i,ticker)
            ticker_history = self.download_data(ticker)

            ticker_history['Ticker'] = ticker
            ticker_history['Year'] = ticker_history.index.year.astype(int)
            ticker_history['Month'] = ticker_history.index.month.astype(int)
            ticker_history['Weekday'] = ticker_history.index.weekday.astype(int)
            ticker_history['Date'] = pd.to_datetime(ticker_history.index.date)

            # sleep 1 sec between downloads - not to overload the API server
            time.sleep(1)
            if stocks_df.empty:
                stocks_df = ticker_history
            else:
                stocks_df = pd.concat([stocks_df, ticker_history], ignore_index=True)
        self.ticker_df = stocks_df

    def save_file(self):
        filepath = os.path.join(self.config.root_dir, 'tickers_df.parquet')
        self.ticker_df.to_parquet(filepath, engine='fastparquet',compression='brotli', index=False)

    def load(self):
        filepath = os.path.join(self.config.root_dir, 'tickers_df.parquet')
        self.ticker_df = pd.read_parquet(filepath)