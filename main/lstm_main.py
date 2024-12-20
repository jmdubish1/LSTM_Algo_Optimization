import pandas as pd
import tensorflow as tf
from nn_tools.process_handler import ProcessHandler
from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
from analysis_tools.param_chooser import AlgoParamResults
from data_tools.data_trade_tools import TradeData
from nn_tools.class_lstm_model_tools import ClassLstmModel
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
    'model_type': 'classification_lstm',
    'strategy': 'Double_Candle',
    'security': 'NQ',
    'other_securities': ['RTY', 'YM', 'ES'],  #, 'GC', 'CL'],
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
    'total_param_sets': 'EvenMoney',
    'chosen_params': {'Bull': [1753, 1822],
                      'Bear': []},
    'percentiles': {'loss': 45,
                    'win': 50},
    'classes': ['lg_loss', 'sm_loss', 'sm_win', 'lg_win']
}

lstm_model_dict = {
    'intra_lookback': 16,
    'daily_lookback': 18,
    'plot_live': True,
    'epochs': {'Bull': 55,
               'Bear': 80},
    'batch_size': 16,
    'buffer_batch_num': 500,  # Exact number of trades to pull at once
    'max_accuracy': .5,
    'lstm_i1_nodes': 384,
    'lstm_i2_nodes': 256,
    'dense_m1_nodes': 256,
    'dense_wl1_nodes': 192,
    'dense_pl1_nodes': 160,
    'adam_optimizer': .00002,
    'prediction_runs': 50,
    'opt_threshold': {'Bull': .50,
                      'Bear': .50},
    'temperature': {'Bull': 1.0,
                    'Bear': 1.0}
}

train_dict = {
    'predict_tf': True,
    'retrain_tf': False,
    'use_prev_period_model': False,
    'train_bad_params': True,
    'over_sample_y': True
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
    param_chooser.run_param_chooser(even_money=True)

    for ph.side in setup_dict['sides']:
        trade_data = TradeData(ph)
        valid_params = param_chooser.valid_param_list(train_bad_params)

        for ph.paramset_id in [911]:
            print(f'Working Paramset: {ph.paramset_id}')
            MktDataWorking(ph)
            if ph.setup_params.model_type == 'classification_lstm':
                lstm_model = ClassLstmModel(ph, lstm_model_dict)
            else:
                pass
            ModelOutputData(ph)

            print(f'Testing Dates: \n'
                  f'...{ph.test_dates}')
            for ind, test_date in enumerate(ph.test_dates):
                print(f'Modelling {test_date}')
                trade_data.set_dates(test_date)
                lstm_data = LstmData(ph)
                trade_data.create_working_df()
                save_handler.set_model_train_paths()
                trade_data.separate_train_test()
                lstm_model.get_loss_penalty_matrix()

                ph.decide_model_to_train(test_date, use_prev_period_model)
                ph.decide_load_prior_model()
                load_scalers = ph.decide_load_scalers()

                if ph.train_modeltf or predict_tf:
                    lstm_data.prep_train_test_data(load_scalers, train_dict['over_sample_y'])
                    param_chooser.adj_lstm_training_nodes(ind, model_increase=.10)
                    if ph.train_modeltf:
                        ph.ph_train_model(ind, train_dict['over_sample_y'])

                    ph.save_handler.load_current_test_date_model()
                    model_output_data = ModelOutputData(ph)
                    model_output_data.predict_data()
                    predicted_data = model_output_data.prediction_analysis(include_small_tf=True)
                    save_handler.save_all_prediction_data(test_date, predicted_data, include_small_tf=True)
                    ph.reset_gpu_memory()
                    # breakpoint()


if __name__ == '__main__':
    main()