import pandas as pd
import numpy as np
import tensorflow as tf
from nn_tools.process_handler import ProcessHandler
from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
from analysis_tools.param_chooser import AlgoParamResults
from data_tools.data_trade_tools import TradeData
from nn_tools.model_tools import LstmOptModel
from data_tools.data_prediction_tools import ModelOutputData
from nn_tools.save_handler import SaveHandler


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

pd.options.mode.chained_assignment = None  # default='warn'


setup_dict = {
    'strategy': 'Double_Candle',
    'model_type': 'LSTM',
    'security': 'NQ',
    'other_securities': ['RTY', 'YM'], #'RTY', 'ES', 'YM', 'GC', 'CL'],
    'sides': ['Bull'],
    'time_frame_test': '15min',
    'time_frame_train': '15min',
    'time_length': '20years',
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',
    'trade_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR',
    'start_train_date': '2010-04-01',
    'final_test_date': '2024-04-01',
    'start_hour': 6,
    'start_minute': 30,
    'test_period_days': 7*13,
    'years_to_train': 3,
    'sample_percent': 1,
    'total_param_sets': 289,
    'chosen_params': {'Bull': [104, 108, 12, 120, 14, 188, 234, 24, 252, 44, 76],
                      'Bear': [100, 118, 124, 160, 26, 32, 34, 40, 74]},
}

lstm_model_dict = {
    'period_lookback': 12,
    'plot_live': False,
    'epochs': {'Bull': 150,
               'Bear': 150},
    'batch_size': 32,
    'max_accuracy': .96,
    'lstm_i1_nodes': 48,
    'lstm_i2_nodes': 32,
    'dense_m1_nodes': 32,
    'dense_wl1_nodes': 12,
    'dense_pl1_nodes': 12,
    'adam_optimizer': .00005,
    'prediction_runs': 1,
    'opt_threshold': {'Bull': .40,
                      'Bear': .40},
    'temperature': {'Bull': 1.0,
                    'Bear': 1.0}
}

train_dict = {
    'predict_tf': True,
    'retrain_tf': False,
    'use_prev_period_model': True,
    'train_bad_params': True
}

def main():
    predict_tf = train_dict['predict_tf']
    retrain_tf = train_dict['retrain_tf']
    use_prev_period_model = train_dict['use_prev_period_model']
    train_bad_params = train_dict['train_bad_params']

    ph = ProcessHandler(setup_params=setup_dict)
    mkt_data = MktDataSetup(ph)
    save_handler = SaveHandler(ph)
    param_chooser = AlgoParamResults(ph)
    param_chooser.run_param_chooser()

    for ph.side in setup_dict['sides']:
        trade_data = TradeData(ph)
        valid_params = param_chooser.valid_param_list(train_bad_params)

        for ph.paramset_id in valid_params[0]:
            mktdata_working = MktDataWorking(ph)
            lstm_model = LstmOptModel(ph, lstm_model_dict)
            model_output_data = ModelOutputData(ph)

            print(f'Testing Dates: \n'
                  f'...{ph.test_dates}')
            for i in range(len(ph.test_dates)):
                test_date = ph.test_dates[i]
                trade_data.separate_train_test(ph)
                trade_data.create_working_df()
                save_handler.set_model_train_paths(test_date)
                trade_data.separate_train_test(test_date)

                ph.decide_model_to_train()



if __name__ == '__main__':
    main()