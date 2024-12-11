from arch import arch_model
import numpy as np
import pandas as pd
import analysis_tools.fractional_difference_optimization as fdo


def main():
    data_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data'
    strat_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR'
    save_loc = f'{strat_loc}\\agg_data'
    securities = ['NQ', 'GC', 'CL', 'NQ', 'RTY', 'ES', 'YM']
    time_frames = ['daily', '15min', '5min']

    cpi_file = f'{data_loc}\\inflation_adjusted\\CPIAUCSL_seasonally_adj.xlsx'
    cpi_df = pd.read_excel(cpi_file, sheet_name='Total_inflation')
    cpi_df['observation_date'] = pd.to_datetime(cpi_df['observation_date'])

    weights = {'w_corr': .75,
               'w_d': .25}

    window_dict = {'Close': list(range(25, 126, 25)),
                   'Vol':  list(range(15, 26, 5)),
                   'OpenInt': list(range(15, 26, 5))}

    plot_opt_window = False
    plot_opt_d = True
    optimize_d_window = True

    for sec in securities:
        for timef in time_frames:
            data_end = f'{sec}_{timef}_20240505_20040401.txt'
            df = fdo.load_prep_data(data_loc, data_end)