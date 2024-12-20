from __future__ import annotations

import gc
import os
import pandas as pd
from datetime import timedelta
import data_tools.general_tools as gt
import cProfile
import pstats
import io
import tensorflow as tf
from numba import cuda

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_tools.data_trade_tools import TradeData
    from analysis_tools.param_chooser import AlgoParamResults
    from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
    from nn_tools.save_handler import SaveHandler
    from data_tools.data_prediction_tools import ModelOutputData
    from data_tools.data_mkt_lstm_tools import LstmData


class SetupParams:
    def __init__(self, setup_dict):
        self.strategy = setup_dict['strategy']
        self.model_type = setup_dict['model_type']
        self.security = setup_dict['security']
        self.other_securities = setup_dict['other_securities']
        self.sides = setup_dict['sides']
        self.time_frame_test = setup_dict['time_frame_test']
        self.time_frame_train = setup_dict['time_frame_train']
        self.time_len = setup_dict['time_length']
        self.data_loc = setup_dict['data_loc']
        self.trade_dat_loc = setup_dict['trade_dat_loc']
        self.start_train_date = pd.to_datetime(setup_dict['start_train_date'], format='%Y-%m-%d')
        self.final_test_date = pd.to_datetime(setup_dict['final_test_date'], format='%Y-%m-%d')
        self.start_hour = setup_dict['start_hour']
        self.start_minute = setup_dict['start_minute']
        self.test_period_days = setup_dict['test_period_days']
        self.years_to_train = setup_dict['years_to_train']
        self.sample_percent = setup_dict['sample_percent']
        self.total_param_sets = setup_dict['total_param_sets']
        self.chosen_params = setup_dict['chosen_params']
        self.classes = setup_dict['classes']
        self.num_y_cols = len(self.classes)
        self.percentiles = setup_dict['percentiles']



class ProcessHandler:
    def __init__(self, setup_params: dict):
        self.setup_params = SetupParams(setup_params)
        self.lstm_model = None
        self.save_handler: "SaveHandler"
        self.mkt_setup: "MktDataSetup"
        self.mktdata_working: "MktDataWorking"
        self.lstm_data: "LstmData"
        self.trade_data: "TradeData"
        self.param_chooser: "AlgoParamResults"
        self.model_output_data: "ModelOutputData"

        self.test_dates = self.get_test_dates()

        self.train_modeltf = True
        self.retraintf = False
        self.predict_datatf = True
        self.prior_traintf = False
        self.load_current_model = False
        self.load_previous_model = False
        self.previous_train_path = None
        self.side = str()
        self.paramset_id = int()

    def get_test_dates(self):
        """Gets a list of all test_date's to train. This should go in another class (possibly processHandler)"""
        end_date = pd.to_datetime(self.setup_params.final_test_date, format='%Y-%m-%d')
        end_date = gt.ensure_friday(end_date)
        start_date = end_date - timedelta(weeks=self.setup_params.years_to_train*52)

        test_dates = []
        current_date = start_date
        while current_date <= end_date:
            test_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=self.setup_params.test_period_days)

        return test_dates

    def decide_model_to_train(self, test_date, use_previous_model):
        current_model_exists = os.path.exists(f'{self.save_handler.model_save_path}\\model.keras')
        previous_model_exists = os.path.exists(f'{self.save_handler.previous_model_path}\\model.keras')
        self.prior_traintf = False
        self.load_current_model = False
        self.load_previous_model = False

        if current_model_exists:
            print(f'Retraining Model: {self.save_handler.model_save_path}')
            self.prior_traintf = True
            self.load_current_model = True
            self.previous_train_path = self.save_handler.model_save_path

            if not self.retraintf:
                print(f'Predicting only: {self.save_handler.previous_model_path}')
                self.train_modeltf = False

        elif previous_model_exists and use_previous_model:
            print(f'Training model from previous model: {self.save_handler.previous_model_path}')
            self.prior_traintf = True
            self.train_modeltf = True
            self.load_previous_model = True
            self.previous_train_path = self.save_handler.previous_model_path

        else:
            print(f'Training New Model...')
            self.train_modeltf = True
            self.prior_traintf = False
        print(f'Training Model: \n...Param: {self.paramset_id} \n...Side: {self.side} \n...Test Date: {test_date}')

    def decide_load_prior_model(self):
        if self.prior_traintf:
            print(f'Loading Prior Model: {self.previous_train_path}')
            if self.load_current_model:
                self.save_handler.load_current_test_date_model()
            elif self.load_previous_model:
                self.save_handler.load_prior_test_date_model()

    def decide_load_scalers(self):
        load_scalers = False
        if self.prior_traintf:
            load_scalers = True
            self.save_handler.load_scalers(self.retraintf)

        else:
            print('Creating New Scalers')

        return load_scalers

    def ph_train_model(self, ind, randomize_tf):
        if not self.prior_traintf:
            self.lstm_model.build_compile_model()
        else:
            print(f'Loaded Previous Model')
        self.lstm_model.train_model(randomize_tf)

        self.save_handler.save_model(ind)
        self.save_handler.save_scalers()

    def reset_gpu_memory(self):
        try:
            tf.keras.backend.clear_session()
            gc.collect()

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            print("GPU memory reset successfully.")
        except Exception as e:
            print(f"Error during GPU memory reset: {e}")

