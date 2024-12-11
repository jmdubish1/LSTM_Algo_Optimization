import numpy as np
import pandas as pd
import data_tools.math_tools as mt
import data_tools.general_tools as gt
import data_tools.data_trade_tools as tdt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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

    def prep_train_test_data(self, load_scalers):
        self.set_x_train_test_datasets()
        self.scale_x_data(load_scalers)
        self.scale_y_pnl_data(load_scalers)
        self.onehot_y_wl_data()

    def set_x_train_test_datasets(self):
        print('\nBuilding X-Train and Test Datasets')
        mktw = self.ph.mktdata_working
        self.x_train_intra = (
            mktw.intra_working)[mktw.intra_working['DateTime'].dt.date.isin(self.ph.trade_data.train_dates)]

        self.x_test_intra = (
            mktw.intra_working)[mktw.intra_working['DateTime'].dt.date.isin(self.ph.trade_data.test_dates)]

        train_dates = (
                self.ph.trade_data.add_to_daily_dates(self.ph.lstm_model.daily_len, train=True) +
                self.ph.trade_data.train_dates)
        self.x_train_daily = mktw.daily_working[mktw.daily_working['DateTime'].dt.date.isin(train_dates)]

        test_dates = (
                self.ph.trade_data.add_to_daily_dates(self.ph.lstm_model.daily_len, train=False) +
                self.ph.trade_data.test_dates)
        self.x_test_daily = mktw.daily_working[mktw.daily_working['DateTime'].dt.date.isin(test_dates)]

    def scale_x_data(self, load_scalers):
        print('\nScaling X Data')
        self.x_train_intra.iloc[:, 1:] = self.x_train_intra.iloc[:, 1:].astype('float32')
        self.x_test_intra.iloc[:, 1:] = self.x_test_intra.iloc[:, 1:].astype('float32')
        self.x_train_daily.iloc[:, 1:] = self.x_train_daily.iloc[:, 1:].astype('float32')
        self.x_test_daily.iloc[:, 1:] = self.x_test_daily.iloc[:, 1:].astype('float32')

        if self.ph.train_modeltf:
            if load_scalers:
                print('Using Previously Loaded X-Scalers')

            else:
                print('Creating New X-Scalers')
                self.intra_scaler = StandardScaler()
                self.intra_scaler.fit(self.x_train_intra.iloc[:, 1:].values)

                self.daily_scaler = StandardScaler()
                self.daily_scaler.fit(self.x_train_daily.iloc[:, 1:].values)

        self.x_train_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_train_intra.iloc[:, 1:].values)
        self.x_test_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_test_intra.iloc[:, 1:].values)

        self.x_train_daily.iloc[:, 1:] = self.daily_scaler.transform(self.x_train_daily.iloc[:, 1:].values)
        self.x_test_daily.iloc[:, 1:] = self.daily_scaler.transform(self.x_test_daily.iloc[:, 1:].values)

    def scale_y_pnl_data(self, load_scalers):
        print('\nScaling Y-pnl Data')

        self.y_train_pnl_df = self.ph.trade_data.y_train_df.iloc[:, :2]
        self.y_train_pnl_df.iloc[:, 1] = self.y_train_pnl_df.iloc[:, 1].astype('float32')

        self.y_test_pnl_df = self.ph.trade_data.y_test_df.iloc[:, :2]
        self.y_test_pnl_df.iloc[:, 1] = self.y_test_pnl_df.iloc[:, 1].astype('float32')

        if self.ph.train_modeltf:
            if load_scalers:
                print('Using Previously Loaded PnL-Scalers')

            else:
                print('Creating New PnL-Scalers')
                self.y_pnl_scaler = StandardScaler()
                self.y_pnl_scaler.fit(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))

        pnl_scaled = self.y_pnl_scaler.transform(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))
        self.y_train_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

        if len(self.y_test_pnl_df) > 0:
            pnl_scaled = self.y_pnl_scaler.transform(self.y_test_pnl_df.iloc[:, 1].values.reshape(-1, 1))
            self.y_test_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

    def onehot_y_wl_data(self):
        print('\nOnehotting WL Data')
        self.y_wl_onehot_scaler = OneHotEncoder(sparse_output=False)

        self.y_train_wl_df = self.ph.trade_data.y_train_df.iloc[:, [0, 2]]

        wl_dat = self.y_train_wl_df.iloc[:, 1].values
        wl_dat = self.y_wl_onehot_scaler.fit_transform(wl_dat.reshape(-1, 1))

        self.y_train_wl_df[['Loss', 'Win']] = wl_dat
        self.y_train_wl_df.drop('Win_Loss', inplace=True, axis=1)

        self.y_test_wl_df = self.ph.trade_data.y_test_df.iloc[:, [0, 2]]
        if len(self.y_test_wl_df) > 0:
            wl_dat = self.y_test_wl_df.iloc[:, 1].values
            wl_dat = self.y_wl_onehot_scaler.transform(wl_dat.reshape(-1, 1))
            self.y_test_wl_df[['Loss', 'Win']] = wl_dat
            self.y_test_wl_df.drop('Win_Loss', inplace=True, axis=1)

    def onehot_y_wl_label_encoder(self):
        print('\nOnehotting WL Data')
        self.y_wl_onehot_scaler = LabelEncoder()

        self.y_train_wl_df = self.ph.trade_data.y_train_df.iloc[:, [0, 2]]

        wl_dat = self.y_train_wl_df.iloc[:, 1].values
        wl_dat = self.y_wl_onehot_scaler.fit_transform(wl_dat.reshape(-1, 1))

        self.y_train_wl_df[['Loss', 'Win']] = wl_dat
        self.y_train_wl_df.drop('Win_Loss', inplace=True, axis=1)

        self.y_test_wl_df = self.ph.trade_data.y_test_df.iloc[:, [0, 2]]
        if len(self.y_test_wl_df) > 0:
            wl_dat = self.y_test_wl_df.iloc[:, 1].values
            wl_dat = self.y_wl_onehot_scaler.transform(wl_dat.reshape(-1, 1))
            self.y_test_wl_df[['Loss', 'Win']] = wl_dat
            self.y_test_wl_df.drop('Win_Loss', inplace=True, axis=1)

    """-------------------------------------------Data Generator-----------------------------------------------------"""
    def grab_prep_trade_lstm(self, y_pnl_df, y_wl_df, x_intraday, x_daily, train_ind, daily_len, intra_len):
        while True:
            try:
                trade_dt = y_pnl_df.iloc[train_ind]['DateTime']

                x_daily_input = x_daily[x_daily['DateTime'] < trade_dt]
                x_daily_input = x_daily_input.iloc[-daily_len:, 1:].to_numpy()
                x_daily_input = gt.pad_to_length(x_daily_input, daily_len)

                x_intra_input = x_intraday[(x_intraday['DateTime'] <= trade_dt) &
                                           (x_intraday['DateTime'] >=
                                            trade_dt.replace(hour=self.ph.setup_params.start_hour,
                                                             minute=self.ph.setup_params.start_minute))]

                x_intra_input = x_intra_input.iloc[-intra_len:, 1:].to_numpy()
                x_intra_input = gt.pad_to_length(x_intra_input, intra_len)

                y_pnl_input = np.array([y_pnl_df.iloc[train_ind, 1]])

                y_wl_input = y_wl_df.iloc[train_ind, 1:].values

                yield x_daily_input, x_intra_input, y_pnl_input, y_wl_input

            except StopIteration:
                break

    def create_batch_input_lstm(self, train_inds, train=True):
        daily_len = self.ph.lstm_model.daily_len
        intra_len = self.ph.lstm_model.intra_len
        y_pnl_df = self.y_train_pnl_df if train else self.y_test_pnl_df
        y_wl_df = self.y_train_wl_df if train else self.y_test_wl_df
        x_intraday = self.x_train_intra if train else self.x_test_intra
        x_daily = self.x_train_daily if train else self.x_test_daily

        while True:
            x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = [], [], [], []

            try:
                for train_ind in train_inds:
                    x_day, x_intra, y_pnl, y_wl = next(self.grab_prep_trade_lstm(y_pnl_df, y_wl_df, x_intraday, x_daily,
                                                                                 train_ind, daily_len, intra_len))
                    x_day_arr.append(x_day)
                    x_intra_arr.append(x_intra)
                    y_pnl_arr.append(y_pnl)
                    y_wl_arr.append(y_wl)

                x_day_arr = tf.convert_to_tensor(x_day_arr, dtype=tf.float32)
                x_intra_arr = tf.convert_to_tensor(x_intra_arr, dtype=tf.float32)
                y_pnl_arr = tf.convert_to_tensor(y_pnl_arr, dtype=tf.float32)
                y_wl_arr = tf.convert_to_tensor(y_wl_arr, dtype=tf.float32)

                yield (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

            except StopIteration:
                if x_day_arr:
                    x_day_arr = tf.convert_to_tensor(x_day_arr, dtype=tf.float32)
                    x_intra_arr = tf.convert_to_tensor(x_intra_arr, dtype=tf.float32)
                    y_pnl_arr = tf.convert_to_tensor(y_pnl_arr, dtype=tf.float32)
                    y_wl_arr = tf.convert_to_tensor(y_wl_arr, dtype=tf.float32)

                yield (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

            break

    def grab_prep_trade_mdn(self, y_pnl_df, y_wl_df, x_intraday, x_daily, train_ind, daily_len, intra_len):
        while True:
            try:
                trade_dt = y_pnl_df.iloc[train_ind]['DateTime']

                x_daily_input = x_daily[x_daily['DateTime'] < trade_dt]
                x_daily_input = x_daily_input.iloc[-daily_len:, 1:].to_numpy()
                x_daily_input = gt.pad_to_length(x_daily_input, daily_len)

                x_intra_input = x_intraday[(x_intraday['DateTime'] <= trade_dt) &
                                           (x_intraday['DateTime'] >=
                                            trade_dt.replace(hour=self.ph.setup_params.start_hour,
                                                             minute=self.ph.setup_params.start_minute))]

                x_intra_input = x_intra_input.iloc[-intra_len:, 1:].to_numpy()
                x_intra_input = gt.pad_to_length(x_intra_input, intra_len)

                y_pnl_input = np.array([y_pnl_df.iloc[train_ind, 1]])

                y_wl_input = y_wl_df.iloc[train_ind, 1:].values

                yield x_daily_input, x_intra_input, y_pnl_input, y_wl_input

            except StopIteration:
                break

    def create_batch_input_mdn(self, train_inds, train=True):
        daily_len = self.ph.lstm_model.daily_len
        intra_len = self.ph.lstm_model.intra_len
        y_pnl_df = self.y_train_pnl_df if train else self.y_test_pnl_df
        y_wl_df = self.y_train_wl_df if train else self.y_test_wl_df
        x_intraday = self.x_train_intra if train else self.x_test_intra
        x_daily = self.x_train_daily if train else self.x_test_daily

        while True:
            x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = [], [], [], []

            try:
                for train_ind in train_inds:
                    x_day, x_intra, y_pnl, y_wl = next(self.grab_prep_trade_lstm(y_pnl_df, y_wl_df, x_intraday, x_daily,
                                                                                 train_ind, daily_len, intra_len))
                    x_day_arr.append(x_day)
                    x_intra_arr.append(x_intra)
                    y_pnl_arr.append(y_pnl)
                    y_wl_arr.append(y_wl)

                x_day_arr = tf.convert_to_tensor(x_day_arr, dtype=tf.float32)
                x_intra_arr = tf.convert_to_tensor(x_intra_arr, dtype=tf.float32)
                y_pnl_arr = tf.convert_to_tensor(y_pnl_arr, dtype=tf.float32)
                y_wl_arr = tf.convert_to_tensor(y_wl_arr, dtype=tf.float32)

                yield (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

            except StopIteration:
                if x_day_arr:
                    x_day_arr = tf.convert_to_tensor(x_day_arr, dtype=tf.float32)
                    x_intra_arr = tf.convert_to_tensor(x_intra_arr, dtype=tf.float32)
                    y_pnl_arr = tf.convert_to_tensor(y_pnl_arr, dtype=tf.float32)
                    y_wl_arr = tf.convert_to_tensor(y_wl_arr, dtype=tf.float32)

                yield (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

            break
