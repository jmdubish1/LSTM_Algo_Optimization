import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import data_tools.general_tools as gt
import sys
from data_tools.math_tools import find_percentile_for_percent_sum, clip_array

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class TradeData:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.trade_data = self

        self.data_loc = str
        self.trade_df = pd.DataFrame()
        self.param_df = pd.DataFrame()
        self.analysis_df = pd.DataFrame()

        self.working_df = pd.DataFrame()
        self.y_train_df = pd.DataFrame()

        self.train_dates = []
        self.curr_test_date = None
        self.start_period_test_date = None
        self.test_dates = []
        self.y_test_df = pd.DataFrame()
        self.valid_params = []

        self.prep_trade_data()

    def prep_trade_data(self):
        self.get_trade_data()
        self.set_pnl()
        self.trade_df['DateTime'] = pd.to_datetime(self.trade_df['DateTime'])

    def set_feather_loc(self):
        self.data_loc = (f'{self.ph.setup_params.trade_dat_loc}\\{self.ph.setup_params.security}\\'
                         f'{self.ph.setup_params.time_frame_test}\\{self.ph.setup_params.time_frame_test}'
                         f'_test_{self.ph.setup_params.time_len}')

    def get_trade_data(self):
        print('\nGetting Trade Data')
        self.set_feather_loc()
        self.trade_df = pd.read_feather(f'{self.data_loc}\\{self.ph.setup_params.security}_'
                                        f'{self.ph.setup_params.time_frame_test}_Double_Candle_'
                                        f'{self.ph.setup_params.total_param_sets}_trades.feather')
        self.param_df = pd.read_feather(f'{self.data_loc}\\{self.ph.setup_params.security}_'
                                        f'{self.ph.setup_params.time_frame_test}_Double_Candle_'
                                        f'{self.ph.setup_params.total_param_sets}_params.feather')
        self.trade_df['DateTime'] = gt.adjust_datetime(self.trade_df['DateTime'])
        self.trade_df = (
            self.trade_df)[self.trade_df['DateTime'].dt.date >=
                           pd.to_datetime(self.ph.setup_params.start_train_date).date()]

    def set_dates(self, test_date):
        self.curr_test_date = pd.to_datetime(test_date)
        self.start_period_test_date = self.curr_test_date - timedelta(self.ph.setup_params.test_period_days)

    def set_pnl(self):
        self.trade_df['PnL'] = np.where(self.trade_df['side'] == 'Bear',
                                        self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                        self.trade_df['exitPrice'] - self.trade_df['entryPrice'])
        self.trade_df['Win_Loss'] = np.where(self.trade_df['PnL'] > 0, 'Win', 'Loss')
        self.trade_df['PnL'] = self.trade_df['PnL']/self.trade_df['entryPrice'] * 100

    def set_pnl_label(self):
        low_percentile = self.ph.setup_params.percentiles['loss']
        high_percentile = self.ph.setup_params.percentiles['win']
        self.working_df['Label'] = np.empty(len(self.working_df), dtype=object)

        temp_df = self.working_df[self.working_df['DateTime'] <= self.start_period_test_date]
        print(temp_df)
        breakpoint()
        pnl_arr = temp_df['PnL'].values
        pnl_arr = clip_array(pnl_arr, 5, 95)
        threshold_low = -find_percentile_for_percent_sum(-pnl_arr[pnl_arr < 0], low_percentile)
        threshold_high = find_percentile_for_percent_sum(pnl_arr[pnl_arr > 0], high_percentile)

        if len(self.ph.setup_params.classes) == 3:
            conds = [(self.working_df['PnL'] <= threshold_low),
                     (self.working_df['PnL'] < threshold_high) &
                     (self.working_df['PnL'] > threshold_low),
                     (self.working_df['PnL'] >= threshold_high)]
            labels = ['lg_loss', 'skip', 'lg_win']
            default = 'skip'
            self.working_df.loc[self.working_df.index, 'Label'] = np.select(conds, labels, default=default)

        else:
            conds = [(self.working_df['PnL'] <= threshold_low),
                     (self.working_df['PnL'] > threshold_low) & (self.working_df['PnL'] <= 0),
                     (self.working_df['PnL'] < threshold_high) & (self.working_df['PnL'] > 0),
                     (self.working_df['PnL'] >= threshold_high)]
            labels = ['lg_loss', 'sm_loss', 'sm_win', 'lg_win']
            default = 'sm_loss'
            self.working_df.loc[self.working_df.index, 'Label'] = np.select(conds, labels, default=default)

        if 'Win_Loss' in self.working_df.columns:
            self.working_df.drop(columns=['Win_Loss'], inplace=True)

    def create_working_df(self):
        print('\nCreating Trades Work Df')
        self.working_df = self.trade_df[(self.trade_df['paramset_id'] == self.ph.paramset_id) &
                                        (self.trade_df['side'] == self.ph.side)]
        self.set_pnl_label()

    def separate_train_test(self):
        print('\nSeparating Train-Test')
        self.subset_test_period()
        train_df = self.working_df[self.working_df['DateTime'] <= self.start_period_test_date]
        self.train_dates = list(np.unique(train_df['DateTime'].dt.date))
        self.y_train_df = train_df

        test_df = self.working_df[(self.working_df['DateTime'] > self.start_period_test_date)]
        self.test_dates = list(np.unique(test_df['DateTime'].dt.date))
        self.y_test_df = test_df

    def add_to_daily_dates(self, num_dates, train=True):
        add_days = []
        if train:
            initial_dates = self.train_dates[0]
        else:
            initial_dates = self.test_dates[0]

        for i in list(range(1, num_dates*2))[::-1]:
            add_days.append((initial_dates - timedelta(days=i)))

        return add_days

    def subset_test_period(self):
        self.working_df = self.working_df[(self.working_df['DateTime'].dt.date <= self.curr_test_date.date())]
        if self.ph.setup_params.model_type == 'mdn_lstm':
            self.working_df = self.working_df[['DateTime', 'PnL']].reset_index(drop=True)
            self.analysis_df = self.working_df.copy(deep=True)
            self.analysis_df['Algo_PnL'] = np.array(self.analysis_df['PnL'])
        else:
            self.analysis_df = self.working_df.copy(deep=True)
            self.working_df = self.working_df[['DateTime', 'PnL', 'Label']]

            self.analysis_df['Algo_label'] = np.array(self.analysis_df['Label'])
            self.analysis_df['PnL'] = self.analysis_df['PnL'] * self.analysis_df['entryPrice'] / 100

    def clear_trade_data(self):
        self.curr_test_date = None
        self.test_dates = []
        self.y_test_df = None
        self.train_dates = []
        self.y_train_df = None









