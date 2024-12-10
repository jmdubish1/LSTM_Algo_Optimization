import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import data_tools.general_tools as gt

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
                                        f'{self.ph.setup_params.time_frame_test}_Double_Candle_289_trades.feather')
        self.param_df = pd.read_feather(f'{self.data_loc}\\{self.ph.setup_params.security}_'
                                        f'{self.ph.setup_params.time_frame_test}_Double_Candle_289_params.feather')
        self.trade_df['DateTime'] = gt.adjust_datetime(self.trade_df['DateTime'])
        self.trade_df = (
            self.trade_df)[self.trade_df['DateTime'].dt.date >=
                           pd.to_datetime(self.ph.setup_params.start_train_date).date()]

    def set_pnl(self):
        self.trade_df['PnL'] = np.where(self.trade_df['side'] == 'Bear',
                                        self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                        self.trade_df['exitPrice'] - self.trade_df['entryPrice'])
        self.trade_df['Win_Loss'] = np.where(self.trade_df['PnL'] > 0, 'Win', 'Loss')
        self.trade_df['PnL'] = self.trade_df['PnL']/self.trade_df['entryPrice'] * 100

    def create_working_df(self):
        print('\nCreating Trades Work Df')
        self.working_df = self.trade_df[(self.trade_df['paramset_id'] == self.ph.paramset_id) &
                                        (self.trade_df['side'] == self.ph.paramset_id)]

    def separate_train_test(self, curr_test_date):
        print('\nSeparating Train-Test')
        self.curr_test_date = pd.to_datetime(curr_test_date)
        self.subset_test_period()

        self.clip_pnl(low_percentile=1, high_percentile=99)
        self.start_period_test_date = self.curr_test_date - timedelta(self.ph.setup_params.test_period_days)
        train_df = self.working_df[self.working_df['DateTime'] <= self.start_period_test_date]
        self.train_dates = list(np.unique(train_df['DateTime'].dt.date))
        self.y_train_df = train_df

        test_df = self.working_df[(self.working_df['DateTime'] > self.start_period_test_date) &
                                  (self.working_df['DateTime'] <= self.curr_test_date)]
        self.test_dates = list(np.unique(test_df['DateTime'].dt.date))
        self.y_test_df = test_df

    def add_to_daily_dates(self, num_dates, train=True):
        add_days = []
        if train:
            initial_dates = self.train_dates[0]
        else:
            initial_dates = self.test_dates[0]

        for i in list(range(1, num_dates+10))[::-1]:
            add_days.append((initial_dates - timedelta(days=i)))

        return add_days

    def clip_pnl(self, low_percentile=1, high_percentile=99):
        pnl_arr = np.array(self.working_df['PnL'])
        percentile_5 = np.percentile(pnl_arr, low_percentile)
        percentile_95 = np.percentile(pnl_arr, high_percentile)
        self.working_df['PnL'] = np.clip(pnl_arr, percentile_5, percentile_95)

    def subset_test_period(self):
        self.working_df = self.working_df[['DateTime', 'PnL', 'Win_Loss']].reset_index(drop=True)
        self.working_df = self.working_df[(self.working_df['DateTime'].dt.date <= self.curr_test_date.date())]
        self.analysis_df = self.working_df.copy(deep=True)
        self.analysis_df['Algo_PnL'] = np.array(self.analysis_df['PnL'])

    def clear_trade_data(self):
        self.curr_test_date = None
        self.test_dates = []
        self.y_test_df = None
        self.train_dates = []
        self.y_train_df = None

