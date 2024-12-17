import sys

import numpy as np
import pandas as pd
import io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2, L1L2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from nn_tools.custom_callbacks_layers import LivePlotLossesMDN, StopAtAccuracy, MDNLayer
import nn_tools.general_lstm_tools as glt
from data_tools.math_tools import compute_loss_penalty_matrix
from nn_tools.loss_functions import penalized_categorical_crossentropy


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Check device placement for operations
# tf.debugging.set_log_device_placement(True)


class ClassLstmModel:
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
        self.intra_len = lstm_dict['intra_lookback']
        self.daily_len = lstm_dict['daily_lookback']
        self.opt_threshold = self.lstm_dict['opt_threshold'][self.ph.side]
        self.buffer = self.lstm_dict['buffer_batch_num']
        self.penalty_matrix = None

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
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("GPU Details: ", tf.config.list_physical_devices('GPU'))

        self.compile_model()
        self.model.summary()

    def get_class_weights(self):
        y_labels = self.ph.lstm_data.y_train_df['Label']
        uniq_y = np.unique(y_labels)

        label_to_index = {label: idx for idx, label in enumerate(uniq_y)}
        numeric_labels = y_labels.map(label_to_index)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(label_to_index)),
            y=numeric_labels)

        class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}
        class_weight_print = {label: weight for label, weight, in zip(label_to_index.keys(), class_weights)}

        for key, val in class_weight_dict.items():
            if class_weight_dict[key] == class_weight_print['lg_win']:
                class_weight_dict[key] = class_weight_print['lg_win'] * 1.15
                class_weight_print['lg_win'] = class_weight_print['lg_win'] * 1.15

        print(f'Class Weights: {class_weight_print}\n')
        return class_weight_dict

    def get_input_shapes(self):
        daily_shape = (self.daily_len, self.ph.lstm_data.x_train_daily.shape[1] - 1)
        intraday_shape = \
            (self.intra_len, self.ph.lstm_data.x_train_intra.shape[1] - 1)

        self.input_shapes = (daily_shape, intraday_shape)

    def train_model(self, randomize_tf=False):
        epochs = self.epochs
        acc_threshold = self.max_acc

        # lr_scheduler = ReduceLROnPlateau(monitor='loss',
        #                                  factor=0.50,
        #                                  patience=1,
        #                                  min_lr=.00000025,
        #                                  cooldown=3,
        #                                  verbose=2)
        self.model_plot = LivePlotLossesMDN(plot_live=self.lstm_dict['plot_live'])

        train_gen = glt.BufferedBatchSequence(self.ph, self.buffer, train=True, randomize=randomize_tf)
        # train_gen = train_gen.load_full_dataset()
        test_gen = glt.BufferedBatchGenerator(self.ph, self.buffer, train=False, randomize=randomize_tf)
        test_gen = test_gen.load_full_dataset()

        stop_at_accuracy = StopAtAccuracy(accuracy_threshold=acc_threshold)
        class_weights = self.get_class_weights()
        one_cyc_lr_fn = glt.one_cycle_lr(self.lstm_dict['adam_optimizer'], self.epochs)
        one_cyc_lr = tf.keras.callbacks.LearningRateScheduler(one_cyc_lr_fn)
        self.model.fit(train_gen,
                       epochs=epochs,
                       verbose=1,
                       validation_data=test_gen,
                       callbacks=[one_cyc_lr, self.model_plot, stop_at_accuracy],
                       shuffle=False,
                       class_weight=class_weights)
        self.model_plot.save_plot(self.ph.save_handler.data_folder, self.ph.paramset_id)

    def compile_model(self):
        self.optimizer = Adam(self.lstm_dict['adam_optimizer'], clipnorm=1.0)
        # threshold = self.opt_threshold

        self.model = Model(inputs=[self.input_layer_daily, self.input_layer_intraday],
                           outputs=self.win_loss_output)
        penalty_cat_crossentropy = penalized_categorical_crossentropy(self.penalty_matrix)
        self.model.compile(optimizer=self.optimizer,
                           loss=penalty_cat_crossentropy)

        print('New Model Created')
        self.get_model_summary_df()

    def build_lstm_model(self):
        self.get_input_shapes()
        self.input_layer_daily = Input(self.input_shapes[0],
                                       name='daily_input_layer')

        lstm_d1 = LSTM(units=96,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_d1')(self.input_layer_daily)

        drop_d1 = Dropout(0.05, name='drop_d1')(lstm_d1)

        lstm_d2 = LSTM(units=48,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_d2')(drop_d1)

        self.input_layer_intraday = Input(self.input_shapes[1],
                                          name='intra_input_layer')

        lstm_i1 = LSTM(units=self.lstm_dict['lstm_i1_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_i1')(self.input_layer_intraday)

        drop_i1 = Dropout(0.05, name='drop_i1')(lstm_i1)

        lstm_i2 = LSTM(units=self.lstm_dict['lstm_i2_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_i2')(drop_i1)

        merged_lstm = Concatenate(axis=-1,
                                  name='concatenate_timesteps')([lstm_d2, lstm_i2], )

        dense_m1 = Dense(units=self.lstm_dict['dense_m1_nodes'],
                         activation='sigmoid',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.01),
                         name='dense_m1')(merged_lstm)

        drop_i1 = Dropout(0.05, name='drop_m1')(dense_m1)

        dense_wl1 = Dense(units=self.lstm_dict['dense_wl1_nodes'],
                          activation='sigmoid',
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(0.01),
                          name='dense_wl1')(drop_i1)

        # Output layers
        self.win_loss_output = Dense(self.ph.setup_params.num_y_cols,
                                     activation='softmax',
                                     name='wl_class')(dense_wl1)

    def get_model_summary_df(self, printft=False):
        if printft:
            self.model.summary()

        summary_buf = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))

        summary_string = summary_buf.getvalue()
        summary_lines = summary_string.split("\n")

        summary_data = []
        for line in summary_lines:
            split_line = list(filter(None, line.split(" ")))
            if len(split_line) > 1:
                summary_data.append(split_line)

        df_summary = pd.DataFrame(summary_data)
        df_cols = df_summary.iloc[1]
        df_summary = df_summary.iloc[2:].reset_index(drop=True)
        df_summary.columns = df_cols

        self.model_summary = df_summary

    def get_loss_penalty_matrix(self):
        self.penalty_matrix = compute_loss_penalty_matrix(self.ph.trade_data.y_train_df)
        self.ph.trade_data.y_train_df.drop(columns='PnL', inplace=True)
        self.ph.trade_data.y_test_df.drop(columns='PnL', inplace=True)






