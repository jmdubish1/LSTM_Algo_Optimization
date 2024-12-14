import pandas as pd
import numpy as np
import data_tools.general_tools as gt
import data_tools.math_tools as mt
import warnings
from fracdiff import fdiff

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler

warnings.simplefilter(action='ignore', category=FutureWarning)


class MktDataSetup:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.mkt_setup = self
        self.all_secs = [self.ph.setup_params.security] + self.ph.setup_params.other_securities

        self.dailydata_clean = pd.DataFrame()
        self.intradata_clean = pd.DataFrame()
        self.security_df = pd.DataFrame()

        self.load_prep_data('daily')
        self.load_prep_data('intraday')

    def load_prep_data(self, time_frame):
        if time_frame == 'daily':
            data_end = 'daily_20240505_20040401.txt'
        else:
            data_end = f'{self.ph.setup_params.time_frame_train}_20240505_20040401.txt'
        print(f'\nLoading {time_frame} data')

        dfs = []
        for sec in self.all_secs:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{self.ph.setup_params.data_loc}\\{sec}_{data_end}')
            temp_df = gt.convert_date_to_dt(temp_df)

            temp_df = temp_df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Vol']]

            if sec == self.ph.setup_params.security:
                self.security_df = temp_df.copy(deep=True)

            for col in temp_df.columns[1:]:
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            dfs.append(temp_df)

        if time_frame == 'daily':
            self.dailydata_clean = gt.merge_dfs(dfs)
            self.dailydata_clean = gt.set_month_day(self.dailydata_clean, time_frame)
        else:
            self.intradata_clean = gt.merge_dfs(dfs)
            self.intradata_clean = gt.set_month_day(self.intradata_clean, time_frame)


class MktDataWorking:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.mktdata_working = self
        self.daily_working = pd.DataFrame()
        self.intra_working = pd.DataFrame()
        self.ffd_df = pd.read_excel(f'{self.ph.setup_params.trade_dat_loc}\\agg_data\\all_FFD_params.xlsx')

        self.param_id_df = pd.DataFrame()
        self.fastema = int()

        self.prep_working_data('daily')
        self.prep_working_data('intraday')
        # self.subset_start_time()

    def prep_working_data(self, time_frame):
        print(f'Prepping Working Data: {time_frame} ')
        if time_frame == 'daily':
            df = self.ph.mkt_setup.dailydata_clean.copy(deep=True)
            self.fastema = 8
        else:
            df = self.ph.mkt_setup.intradata_clean.copy(deep=True)
            self.get_intra_ema()

        for sec in self.ph.mkt_setup.all_secs:
            df[f'{sec}_ATR_fast'] = mt.create_atr(df, sec, n=4)
            df[f'{sec}_ATR_slow'] = mt.create_atr(df, sec, n=8)
            df[f'{sec}_RSI_k'], df[f'{sec}_RSI_d'] = mt.create_smooth_rsi(df[f'{sec}_Close'], self.fastema)

            df = prep_ema(df, sec, self.fastema)
            df = mt.add_high_low_diff(df, sec)
            df = self.frac_diff(df, sec)
            df = mt.garch_modeling(df, sec)

        df = mt.encode_time_features(df, time_frame)
        df = gt.fill_na_inf(df)

        if time_frame == 'daily':
            self.daily_working = df
        else:
            self.intra_working = df

    def frac_diff(self, df, sec):
        for met in ['Open', 'High', 'Low', 'Close', 'Vol']:
            d_val, ws = self.get_frac_diff_params(met, sec)

            ws = min(ws, 450)
            arr = pd.Series(df[f'{sec}_{met}']).fillna(method='ffill').dropna().values
            out_arr = fdiff(arr, d_val, window=ws, mode='same')
            out_arr = mt.subset_to_first_nonzero(out_arr)
            padding = len(arr) - len(out_arr)
            out_arr = out_arr[~np.isnan(out_arr)]
            out_arr = np.pad(out_arr, (padding, 0), constant_values=0)

            df[f'{sec}_{met}'] = out_arr

        return df

    def get_frac_diff_params(self, met, sec):
        met = 'Close' if met in ['Open', 'High', 'Low'] else met
        df = self.ffd_df.loc[(self.ffd_df['time_frame'] == self.ph.setup_params.time_frame_train) &
                             (self.ffd_df['security'] == sec) &
                             (self.ffd_df['Data'] == met)].reset_index(drop=True)

        d_val, ws = df.loc[0, 'd_val'], df.loc[0, 'window']

        return d_val, ws

    def get_intra_ema(self):
        param_id_df = (
            self.ph.param_chooser.valid_param_df
            [(self.ph.param_chooser.valid_param_df['paramset_id'] == self.ph.paramset_id) &
             (self.ph.param_chooser.valid_param_df['side'] == self.ph.side)]).reset_index(drop=True)
        self.fastema = int(param_id_df.loc[0, 'fastEmaLen'])

    def subset_start_time(self):
        start_time = (
            pd.Timestamp(f'{self.ph.setup_params.start_hour:02}:{self.ph.setup_params.start_minute:02}:00').time())

        self.intra_working['time'] = self.intra_working['DateTime'].dt.time
        subset_df = self.intra_working[
            (self.intra_working['time'] >= start_time) &
            (self.intra_working['time'] <= pd.Timestamp('16:00:00').time())]
        self.intra_working = subset_df.drop(columns=['time'])

        self.intra_working['time'] = self.intra_working['DateTime'].dt.time
        subset_df = self.intra_working[
            (self.intra_working['time'] >= start_time) &
            (self.intra_working['time'] <= pd.Timestamp('16:00:00').time())]
        self.intra_working = subset_df.drop(columns=['time'])


def prep_ema(df, sec, ema_len):
    df[f'{sec}_EMA_{ema_len}'] = mt.calculate_ema_numba(df, f'{sec}_Close', ema_len)
    df[f'{sec}_EMA_Close_{ema_len}'] = (df[f'{sec}_Close'] - df[f'{sec}_EMA_{ema_len}']) / df[f'{sec}_Close'] * 100
    df[f'{sec}_EMA_{ema_len}'] = mt.standardize_ema(df[f'{sec}_EMA_{ema_len}'], ema_len)

    return df






