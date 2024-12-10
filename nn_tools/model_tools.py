import numpy as np
import pandas as pd
import io
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Concatenate, Reshape, GRU)
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import nn_tools.loss_functions as lf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)




class LstmOptModel:
    def __init__(self,
                 process_handler: "ProcessHandler",
                 lstm_dict: dict):
        self.ph = process_handler
        self.ph.lstm_model = self
        self.lstm_dict = lstm_dict

        self.temperature = lstm_dict['temperature'][self.ph.side]
        self.epochs = self.lstm_dict['epochs'][self.ph.side]
        self.batch_s = lstm_dict['batch_size']
        self.max_acc = lstm_dict['max_accuracy']
        self.intra_len = lstm_dict['period_lookback']
        self.opt_threshold = self.lstm_dict['opt_threshold'][self.ph.side]

        self.input_shapes = None
        self.input_layer_daily = None
        self.input_layer_intraday = None
        self.win_loss_output = None
        self.float_output = None

        self.model = None
        self.scheduler = None
        self.optimizer = None

        self.model_plot = None
        self.model_summary = None

    def build_compile_model(self):
        print(f'\nBuilding New Model\n'
              f'Param ID: {self.ph.paramset_id}')
        self.build_lstm_model()
        self.compile_model()
        self.model.summary()

    def get_class_weights(self):
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(self.mkt_data.y_train_wl_df['Win']),
                                             y=self.mkt_data.y_train_wl_df['Win'])

        class_weight_dict_wl = {i: weight for i, weight in enumerate(class_weights)}
        print(f'Class Weights: {class_weight_dict_wl}\n')
        return class_weights



