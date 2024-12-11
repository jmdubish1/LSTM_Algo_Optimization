import pandas as pd
import numpy as np
import tensorflow as tf
from nn_tools.process_handler import ProcessHandler
from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
from analysis_tools.param_chooser import AlgoParamResults
from data_tools.data_trade_tools import TradeData
from nn_tools.basic_lstm_model_tools import LstmOptModel
from data_tools.data_prediction_tools import ModelOutputData
from nn_tools.save_handler import SaveHandler
from data_tools.data_mkt_lstm_tools import LstmData


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

pd.options.mode.chained_assignment = None  # default='warn'


setup_dict = {
    'model_type': 'mdn_lstm',
    'strategy': 'Double_Candle',
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
    'start_hour': 7,
    'start_minute': 30,
    'test_period_days': 7*13,
    'years_to_train': 3,
    'sample_percent': 1,
    'total_param_sets': 289,
    'chosen_params': {'Bull': [104, 108, 12, 120, 14, 188, 234, 24, 252, 44, 76],
                      'Bear': [100, 118, 124, 160, 26, 32, 34, 40, 74]},
}

lstm_model_dict = {
    'intra_lookback': 18,
    'daily_lookback': 24,
    'plot_live': True,
    'epochs': {'Bull': 250,
               'Bear': 250},
    'batch_size': 16,
    'max_accuracy': .96,
    'lstm_i1_nodes': 24,
    'lstm_i2_nodes': 20,
    'dense_m1_nodes': 20,
    'dense_wl1_nodes': 12,
    'dense_pl1_nodes': 12,
    'adam_optimizer': .00005,
    'prediction_runs': 1,
    'opt_threshold': {'Bull': .50,
                      'Bear': .50},
    'temperature': {'Bull': 1.0,
                    'Bear': 1.0}
}

train_dict = {
    'predict_tf': True,
    'retrain_tf': False,
    'use_prev_period_model': True,
    'train_bad_params': False
}

def main():
    predict_tf = train_dict['predict_tf']
    retrain_tf = train_dict['retrain_tf']
    use_prev_period_model = train_dict['use_prev_period_model']
    train_bad_params = train_dict['train_bad_params']

    ph = ProcessHandler(setup_params=setup_dict)
    MktDataSetup(ph)
    save_handler = SaveHandler(ph)
    param_chooser = AlgoParamResults(ph)
    param_chooser.run_param_chooser()

    for ph.side in setup_dict['sides']:
        trade_data = TradeData(ph)
        valid_params = param_chooser.valid_param_list(train_bad_params)

        for ph.paramset_id in valid_params[0:1]:
            MktDataWorking(ph)
            LstmOptModel(ph, lstm_model_dict)
            ModelOutputData(ph)

            print(f'Testing Dates: \n'
                  f'...{ph.test_dates}')
            for ind, test_date in enumerate(ph.test_dates):
                trade_data.set_dates(test_date)
                lstm_data = LstmData(ph)
                trade_data.create_working_df()
                save_handler.set_model_train_paths()
                trade_data.separate_train_test()

                ph.decide_model_to_train(test_date, use_prev_period_model)
                ph.decide_load_prior_model()
                load_scalers = ph.decide_load_scalers()
                ph.lstm_model.modify_op_threshold_temp(ind, mod_thres=True)

                if ph.train_modeltf:
                    lstm_data.prep_train_test_data(load_scalers)
                    param_chooser.adj_lstm_training_nodes()
                    ph.ph_train_model(ind)



if __name__ == '__main__':
    main()