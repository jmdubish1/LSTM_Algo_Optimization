import numpy as np
import pandas as pd
import data_tools.math_tools as mt
import data_tools.general_tools as gt
import data_tools.data_trade_tools as tdt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from datetime import datetime, timedelta
import warnings
import tensorflow as tf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler

warnings.simplefilter(action='ignore', category=FutureWarning)


class LstmData:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.lstm_data = self

        self.x_train_daily = None
        self.x_test_daily = None

        self.x_train_intra = None
        self.x_test_intra = None

        self.y_train_pnl_df = None
        self.y_train_wl_df = None
        self.y_test_pnl_df = None
        self.y_test_wl_df = None

        self.intra_scaler = None
        self.daily_scaler = None
        self.y_pnl_scaler = None
        self.y_wl_onehot_scaler = None


    def set_x_train_test_datasets(self):
        print('\nBuilding X-Train and Test Datasets')
        self.x_train_intra = self.intradata[self.intradata['DateTime'].dt.date.isin(self.trade_data.train_dates)]
        self.x_test_intra = self.intradata[self.intradata['DateTime'].dt.date.isin(self.trade_data.test_dates)]

        train_dates = (
                self.trade_data.add_to_daily_dates(self.lstm_model.daily_len, train=True) + self.trade_data.train_dates)
        self.x_train_daily = self.dailydata[self.dailydata['DateTime'].dt.date.isin(train_dates)]

        test_dates = (
                self.trade_data.add_to_daily_dates(self.lstm_model.daily_len, train=False) + self.trade_data.test_dates)
        self.x_test_daily = self.dailydata[self.dailydata['DateTime'].dt.date.isin(test_dates)]