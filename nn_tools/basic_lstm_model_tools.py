import numpy as np
import pandas as pd
import io
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import nn_tools.loss_functions as lf
from nn_tools.custom_callbacks_layers import CustomDataGenerator, LivePlotLosses, StopAtAccuracy, TemperatureScalingLayer

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
        self.intra_len = lstm_dict['intra_lookback']
        self.daily_len = lstm_dict['daily_lookback']
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
                                             classes=np.unique(self.ph.lstm_data.y_train_wl_df['Win']),
                                             y=self.ph.lstm_data.y_train_wl_df['Win'])

        class_weight_dict_wl = {i: weight for i, weight in enumerate(class_weights)}
        print(f'Class Weights: {class_weight_dict_wl}\n')
        return class_weights

    def get_input_shapes(self):
        daily_shape = (self.daily_len, self.ph.lstm_data.x_train_daily.shape[1] - 1)
        intraday_shape = (self.intra_len, self.ph.lstm_data.x_train_intra.shape[1] - 1)

        self.input_shapes = [daily_shape, intraday_shape]

    def train_model(self, i, previous_train):
        if previous_train:
            if i == 1:
                epochs = int(self.epochs / 4)
                acc_threshold = self.lstm_dict['max_accuracy'] + .01
                self.model.optimizer.learning_rate.assign(self.lstm_dict['adam_optimizer'] / 5)
            else:
                epochs = int(self.epochs / 4)
                acc_threshold = self.lstm_dict['max_accuracy'] + .01
                self.model.optimizer.learning_rate.assign(self.lstm_dict['adam_optimizer'] / 5)
        else:
            epochs = self.epochs
            acc_threshold = self.max_acc

        lr_scheduler = ReduceLROnPlateau(monitor='loss',
                                         factor=0.85,
                                         patience=3,
                                         min_lr=.00000025,
                                         cooldown=3,
                                         verbose=2)
        self.model_plot = LivePlotLosses(plot_live=self.lstm_dict['plot_live'])

        train_data_gen, test_data_gen = self.get_input_datasets()

        stop_at_accuracy = StopAtAccuracy(accuracy_threshold=acc_threshold)

        self.model.fit(train_data_gen,
                       epochs=epochs,
                       verbose=1,
                       validation_data=test_data_gen,
                       callbacks=[lr_scheduler, self.model_plot, stop_at_accuracy],
                       shuffle=False)
        self.model_plot.save_plot(self.ph.save_handler.data_folder, self.ph.paramset_id)

    def compile_model(self):
        self.optimizer = Adam(self.lstm_dict['adam_optimizer'], clipnorm=1.0)
        threshold = self.opt_threshold
        class_weights = self.get_class_weights()
        combined_wl_loss = lf.comb_class_loss(beta=1.5,
                                              opt_threshold=threshold,
                                              class_weights=class_weights)
        npv_fn = lf.negative_predictive_value(threshold)
        auc = lf.weighted_auc(class_weights)
        ppv_fn = lf.positive_predictive_value(threshold)

        self.model = Model(inputs=[self.input_layer_daily, self.input_layer_intraday],
                           outputs=[self.win_loss_output, self.float_output])

        self.model.compile(optimizer=self.optimizer,
                           loss={'wl_class': combined_wl_loss,
                                 'pnl': 'mse'},
                           metrics={'wl_class': [npv_fn,
                                                 ppv_fn,
                                                 auc],
                                    'pnl': 'mse'},
                           loss_weights={'wl_class': 0.60,
                                         'pnl': 0.40})

        print('New Model Created')
        self.get_model_summary_df()

    def build_lstm_model(self):
        self.get_input_shapes()
        self.input_layer_daily = Input(self.input_shapes[0])

        lstm_d1 = LSTM(units=24,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.001),
                       name='lstm_d1')(self.input_layer_daily)

        drop_d1 = Dropout(0.05, name='drop_d1')(lstm_d1)

        lstm_d2 = LSTM(units=16,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_d2')(drop_d1)

        self.input_layer_intraday = Input(self.input_shapes[1])

        lstm_i1 = LSTM(units=self.lstm_dict['lstm_i1_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.001),
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
                                  name='concatenate_timesteps')([lstm_d2, lstm_i2],)

        dense_m1 = Dense(units=self.lstm_dict['dense_m1_nodes'],
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.01),
                         name='dense_m1')(merged_lstm)

        drop_i1 = Dropout(0.05, name='drop_m1')(dense_m1)

        dense_wl1 = Dense(units=self.lstm_dict['dense_wl1_nodes'],
                          activation='sigmoid',
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(0.01),
                          name='dense_wl1')(drop_i1)

        dense_pl1 = Dense(units=self.lstm_dict['dense_pl1_nodes'],
                          activation='tanh',
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(0.01),
                          name='dense_pl1')(drop_i1)

        logit_layer = Dense(2,
                            activation=None,
                            name='logits')(dense_wl1)

        temp_scale_wl = TemperatureScalingLayer(self.temperature,
                                                name='temp_scaling')(logit_layer)

        # Output layers
        self.win_loss_output = Dense(2,
                                     activation='sigmoid',
                                     name='wl_class')(temp_scale_wl)

        self.float_output = Dense(units=1,
                                  activation='tanh',
                                  name='pnl')(dense_pl1)

    def get_generator(self, traintf=True):
        if traintf:
            generator = CustomDataGenerator(self.ph, self.batch_s)
        else:
            generator = CustomDataGenerator(self.ph, self.batch_s, train=False)

        def generator_function():
            for i in range(len(generator)):
                yield generator[i]

        return generator_function

    def get_input_datasets(self):
        daily_shape = self.ph.lstm_data.x_train_daily.shape[1] - 1
        intraday_shape = self.ph.lstm_data.x_train_intra.shape[1] - 1

        output_signature = (
            (tf.TensorSpec(shape=(None, self.daily_len, daily_shape), dtype=tf.float32),
             tf.TensorSpec(shape=(None, self.intra_len, intraday_shape), dtype=tf.float32)),
            {'wl_class': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
             'pnl': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)})

        train_gen = self.get_generator(traintf=True)
        test_gen = self.get_generator(traintf=False)

        train_dataset = tf.data.Dataset.from_generator(
            train_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_generator(
            test_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

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

    def modify_op_threshold_temp(self, ind, mod_thres=True):
        if ind == 0:
            self.opt_threshold = self.lstm_dict['opt_threshold'][self.ph.side]
            self.temperature = self.lstm_dict['temperature'][self.ph.side]
        else:
            opt_df = pd.read_excel(f'{self.ph.save_handler.param_folder}\\best_thresholds.xlsx')

            self.temperature = \
                (opt_df.loc[(opt_df['side'] == self.ph.side) &
                            (opt_df['paramset_id'] == self.ph.paramset_id), 'opt_temp'].values)[0]

            if mod_thres:
                self.opt_threshold = \
                    (opt_df.loc[(opt_df['side'] == self.ph.side) &
                                (opt_df['paramset_id'] == self.ph.paramset_id), 'opt_threshold'].values)[0]
            else:
                self.opt_threshold = self.lstm_dict['opt_threshold'][self.ph.side]

