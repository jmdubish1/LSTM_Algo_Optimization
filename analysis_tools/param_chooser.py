import numpy as np
import pandas as pd
from datetime import timedelta
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class AlgoParamResults:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.param_chooser = self
        self.end_date = pd.to_datetime(self.ph.setup_params.final_test_date, format='%Y-%m-%d')
        self.trade_folder = str()
        self.trades_file = str()
        self.param_file = str()
        self.pnl_df_save_file = str()
        self.params_save_file = str()

        self.trade_df = pd.DataFrame()
        self.param_df = pd.DataFrame()
        self.pnl_df = pd.DataFrame()

        self.best_params_df = pd.DataFrame()
        self.other_params_df = pd.DataFrame()
        self.valid_param_df = pd.DataFrame()

    def set_file_locations(self):
        params = self.ph.setup_params

        self.trade_folder = (f'{params.trade_dat_loc}\\{params.security}\\{params.time_frame_test}\\'
                             f'{params.time_frame_test}_test_{params.time_len}')
        self.trades_file =(f'{self.trade_folder}\\{params.security}_{params.time_frame_test}_'
                           f'{params.strategy}_{params.total_param_sets}_trades.feather')
        self.param_file = (f'{self.trade_folder}\\{params.security}_{params.time_frame_test}_'
                           f'{params.strategy}_{params.total_param_sets}_params.feather')
        self.pnl_df_save_file = f'{self.trade_folder}\\{params.security}_{params.time_frame_test}_all_params.xlsx'

    def load_files(self):
        self.trade_df = pd.read_feather(self.trades_file)
        self.param_df = pd.read_feather(self.param_file)

    def set_pnl(self):
        self.pnl_df = self.trade_df.copy()
        self.pnl_df['PnL'] = np.where(self.trade_df['side'] == 'Bear',
                                      self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                      self.trade_df['exitPrice'] - self.trade_df['entryPrice'])

    def subset_date_agg_pnl(self):
        self.end_date = self.end_date - timedelta(weeks=self.ph.setup_params.years_to_train*52)
        self.pnl_df = self.pnl_df[self.pnl_df['DateTime'] < self.end_date]
        self.pnl_df = self.pnl_df.groupby(['side', 'paramset_id'], as_index=False).agg(
            row_count=('PnL', 'count'),  # Count the number of rows
            total_pnl=('PnL', 'sum'),
            loss_count=('PnL', lambda x: (x < 0).sum()),
            win_count=('PnL', lambda x: (x > 0).sum()),
            avg_pnl=('PnL', 'mean'),
            avg_pnl_neg=('PnL', lambda x: x[x < 0].mean()),
            avg_pnl_pos=('PnL', lambda x: x[x > 0].mean())
        )
        self.pnl_df.rename(columns={'row_count': 'tot_trades',
                                    'total_pnl': 'PnL'}, inplace=True)

        self.pnl_df['win_percent'] = self.pnl_df['win_count']/self.pnl_df['tot_trades']
        self.pnl_df['expected_value'] = ((self.pnl_df['win_percent'] * self.pnl_df['avg_pnl_pos']) +
                                         ((1 - self.pnl_df['win_percent']) * self.pnl_df['avg_pnl_neg']))
        self.pnl_df['max_potential'] = (((self.pnl_df['win_percent'] * self.pnl_df['avg_pnl_pos']) +
                                        (-(1 - self.pnl_df['win_percent']) * self.pnl_df['avg_pnl_neg'])) *
                                        self.pnl_df['tot_trades'])

    def merge_pnl_params(self):
        self.pnl_df = pd.merge(self.pnl_df, self.param_df, on='paramset_id')

    def get_best_params(self):

        best_params = []
        for side in ['Bear', 'Bull']:
            temp_side = self.pnl_df[self.pnl_df['side'] == side]

            temp_side = temp_side.sort_values(['fastEmaLen', 'PnL'], ascending=[True, False])
            temp_ema = temp_side.groupby('fastEmaLen').head(2)
            best_params.append(temp_ema)
            temp_day_ema = temp_side.groupby('dayEma').head(4)
            best_params.append(temp_day_ema)

        best_params = pd.concat(best_params)
        best_params = best_params.sort_values(['PnL'], ascending=False)
        best_params.drop_duplicates(inplace=True)

        self.best_params_df = best_params

    def get_other_params(self):
        other_params = []
        for side in ['Bear', 'Bull']:
            temp_side = self.pnl_df[self.pnl_df['side'] == side]

            temp_side = temp_side.sort_values(['fastEmaLen', 'PnL'], ascending=[True, False]).reset_index(drop=True)
            middle_ind = int(len(temp_side)/2)
            other_params.append(pd.DataFrame(temp_side.iloc[middle_ind + 25]).T)
            for i in range(0, 100, 25):
                other_params.append(pd.DataFrame(temp_side.iloc[middle_ind - i]).T)

        best_params = pd.concat(other_params)
        best_params = best_params.sort_values(['PnL'], ascending=False)
        best_params.drop_duplicates(inplace=True)

        self.other_params_df = best_params

    def save_all_params(self, valid_params, side):
        self.params_save_file = \
            f'{self.trade_folder}\\{side}\\{self.ph.setup_params.security}_{self.ph.setup_params.time_frame_test}'
        os.makedirs(os.path.dirname(self.params_save_file), exist_ok=True)

        self.pnl_df.to_excel(f'{self.pnl_df_save_file}')
        self.best_params_df.to_excel(f'{self.params_save_file}_best_params.xlsx')
        self.other_params_df.to_excel(f'{self.params_save_file}_other_params.xlsx')
        self.valid_param_df = self.pnl_df[self.pnl_df['paramset_id'].isin(valid_params)]
        self.valid_param_df.to_excel(f'{self.params_save_file}_all_analyzed_param.xlsx')

    def run_param_chooser(self):
        self.set_file_locations()
        self.load_files()
        self.set_pnl()
        self.subset_date_agg_pnl()
        self.merge_pnl_params()
        self.get_best_params()
        self.get_other_params()

    def apply_percentiles_to_paramset(self):
        self.pnl_df['percentile'] = 0

        temp_df = self.pnl_df[self.pnl_df['side'] == 'Bear']
        percent = temp_df['tot_trades'].rank(pct=True).values
        self.pnl_df.loc[self.pnl_df['side'] == 'Bear', 'percentile'] = percent

        temp_df = self.pnl_df[self.pnl_df['side'] == 'Bull']
        percent = temp_df['tot_trades'].rank(pct=True).values
        self.pnl_df.loc[self.pnl_df['side'] == 'Bull', 'percentile'] = percent

    def set_lstm_nodes(self):
        self.apply_percentiles_to_paramset()
        for layer in ['lstm_i1_nodes', 'lstm_i2_nodes', 'dense_m1_nodes', 'dense_wl1_nodes', 'dense_pl1_nodes']:
            self.pnl_df[layer] = 0
            self.pnl_df[layer] = self.ph.lstm_model.lstm_dict[layer]
            self.pnl_df.loc[self.pnl_df['percentile'] > 0.5, layer] = (
                    self.pnl_df.loc[self.pnl_df['percentile'] > 0.5, layer] * 1.05)
            self.pnl_df[layer] = self.pnl_df[layer].astype(int)

    def adj_lstm_training_nodes(self):
        self.set_lstm_nodes()
        for layer in ['lstm_i1_nodes', 'lstm_i2_nodes', 'dense_m1_nodes', 'dense_wl1_nodes', 'dense_pl1_nodes']:
            self.ph.lstm_model.lstm_dict[layer] = (
                self.pnl_df.loc[(self.pnl_df['side'] ==
                                 self.ph.side) & (self.pnl_df['paramset_id'] == self.ph.paramset_id), layer].iloc)[0]

    def valid_param_list(self, train_bad_paramsets):
        valid_params = self.get_valid_params(train_bad_paramsets)
        valid_params = np.concatenate((valid_params, self.ph.setup_params.chosen_params[self.ph.side]))

        print(f'Training {len(valid_params)} Valid Params: \n'
              f'{valid_params}')

        return valid_params[::-1]

    def get_valid_params(self, train_bad_paramsets):
        side = self.ph.side
        good_params = np.array(
            self.best_params_df.loc[self.best_params_df['side'] == side, 'paramset_id'])
        other_params = np.array(
            self.other_params_df.loc[self.other_params_df['side'] == side, 'paramset_id'])

        if train_bad_paramsets:
            valid_params = np.concatenate((other_params, good_params))
            valid_params = np.concatenate((valid_params, self.ph.setup_params.chosen_params[side]))
        else:
            valid_params = np.concatenate((good_params, self.ph.setup_params.chosen_params[side]))

        valid_params = sorted(np.unique(valid_params).astype(int))
        self.save_all_params(valid_params, side)

        return valid_params








