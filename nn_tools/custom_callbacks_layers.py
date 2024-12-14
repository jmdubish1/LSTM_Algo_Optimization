import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from nn_tools import general_lstm_tools as glt
import time

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class LivePlotLossesLSTM(Callback):
    def __init__(self, plot_live):
        super(LivePlotLossesLSTM, self).__init__()
        self.plot_live = plot_live
        self.epochs = []

        self.losses = []
        self.wl_class_losses = []
        self.pnl_losses = []
        self.auc_loss = []
        self.wl_class_accs = []
        self.wl_class_accs2 = []

        self.losses_val = []
        self.wl_class_losses_val = []
        self.pnl_losses_val = []
        self.auc_loss_val = []
        self.wl_class_accs_val = []
        self.wl_class_accs_val2 = []

        self.train_loss_line = None
        self.val_loss_line = None
        self.pnl_loss_line = None
        self.val_pnl_loss_line = None
        self.auc_loss_line = None
        self.val_auc_loss_line = None
        self.class_loss_line = None
        self.val_class_loss_line = None
        self.class_npv_line = None
        self.val_class_npv_line = None
        self.class_ppv_line = None
        self.val_class_ppv_line = None

        if self.plot_live:
            plt.ion()
        self.fig, self.axs = plt.subplots(2, 3, figsize=(10, 6))  # Create a 2x2 grid of subplots
        self.fig.tight_layout()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch + 1)  # Accumulate epochs
        self.losses.append(logs.get('loss'))
        self.wl_class_losses.append(logs.get('wl_class_loss'))
        self.pnl_losses.append(logs.get('pnl_loss'))
        self.auc_loss.append(logs.get('wl_class_auc_loss'))
        self.wl_class_accs.append(logs.get('wl_class_npv'))
        self.wl_class_accs2.append(logs.get('wl_class_ppv'))

        self.losses_val.append(logs.get('val_loss'))
        self.wl_class_losses_val.append(logs.get('val_wl_class_loss'))
        self.pnl_losses_val.append(logs.get('val_pnl_loss'))
        self.auc_loss_val.append(logs.get('val_wl_class_auc_loss'))
        self.wl_class_accs_val.append(logs.get('val_wl_class_npv'))
        self.wl_class_accs_val2.append(logs.get('val_wl_class_ppv'))

        if self.train_loss_line is None:
            self.train_loss_line, = self.axs[0, 0].plot([], [], label="Train", marker='.', color='blue')
            self.val_loss_line, = self.axs[0, 0].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[0, 0].set_title("Total Loss")
            self.axs[0, 0].legend()

            self.pnl_loss_line, = self.axs[0, 1].plot([], [], label="Train", marker='.', color='blue')
            self.val_pnl_loss_line, = self.axs[0, 1].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[0, 1].set_title("PnL MSE")

            self.auc_loss_line, = self.axs[0, 2].plot([], [], label="Train", marker='.', color='blue')
            self.val_auc_loss_line, = self.axs[0, 2].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[0, 2].set_title("WL Auc Loss")

            self.class_loss_line, = self.axs[1, 0].plot([], [], label="Train", marker='.', color='blue')
            self.val_class_loss_line, = self.axs[1, 0].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[1, 0].set_title("WL Class Loss")

            self.class_npv_line, = self.axs[1, 1].plot([], [], label="Train", marker='.', color='blue')
            self.val_class_npv_line, = self.axs[1, 1].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[1, 1].set_title("WL Class NPV")

            self.class_ppv_line, = self.axs[1, 2].plot([], [], label="Train", marker='.', color='blue')
            self.val_class_ppv_line, = self.axs[1, 2].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[1, 2].set_title("WL Class PPV")

        self.train_loss_line.set_data(self.epochs, self.losses)
        self.val_loss_line.set_data(self.epochs, self.losses_val)

        self.pnl_loss_line.set_data(self.epochs, self.pnl_losses)
        self.val_pnl_loss_line.set_data(self.epochs, self.pnl_losses_val)

        self.auc_loss_line.set_data(self.epochs, self.auc_loss)
        self.val_auc_loss_line.set_data(self.epochs, self.auc_loss_val)

        self.class_loss_line.set_data(self.epochs, self.wl_class_losses)
        self.val_class_loss_line.set_data(self.epochs, self.wl_class_losses_val)

        self.class_npv_line.set_data(self.epochs, self.wl_class_accs)
        self.val_class_npv_line.set_data(self.epochs, self.wl_class_accs_val)

        self.class_ppv_line.set_data(self.epochs, self.wl_class_accs2)
        self.val_class_ppv_line.set_data(self.epochs, self.wl_class_accs_val2)

        for r in range(2):
            for c in range(3):
                self.axs[r, c].relim()
                self.axs[r, c].autoscale_view()

        if self.plot_live:
            self.fig.canvas.draw()
            plt.pause(0.2)

    def save_plot(self, save_loc, param_id):
        plt.savefig(f'{save_loc}\\param_{param_id}_plot.png', dpi=500)

    def on_train_end(self, logs=None):
        if self.plot_live:
            breakpoint()
            plt.ioff()  # Turn off interactive mode at the end
            plt.close()


class LivePlotLossesMDN(Callback):
    def __init__(self, plot_live):
        super(LivePlotLossesMDN, self).__init__()
        self.plot_live = plot_live
        self.epochs = []

        self.losses = []
        self.losses_val = []

        self.train_loss_line = None
        self.val_loss_line = None

        if self.plot_live:
            plt.ion()
        self.fig, self.axs = plt.subplots(figsize=(8, 4))  # Create a 2x2 grid of subplots
        self.fig.tight_layout()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch + 1)  # Accumulate epochs
        self.losses.append(logs.get('loss'))
        self.losses_val.append(logs.get('val_loss'))

        if self.train_loss_line is None:
            self.train_loss_line, = self.axs.plot([], [], label="Train", marker='.', color='blue')
            self.val_loss_line, = self.axs.plot([], [], label="Val", marker='.', color='darkred')
            self.axs.set_title("Total Loss")
            self.axs.legend()

        self.train_loss_line.set_data(self.epochs, self.losses)
        self.val_loss_line.set_data(self.epochs, self.losses_val)

        self.axs.relim()
        self.axs.autoscale_view()

        if self.plot_live:
            self.fig.canvas.draw()
            plt.pause(0.2)

    def save_plot(self, save_loc, param_id):
        plt.savefig(f'{save_loc}\\param_{param_id}_plot.png', dpi=500)

    def on_train_end(self, logs=None):
        if self.plot_live:
            plt.ioff()  # Turn off interactive mode at the end
            plt.close()


class StopAtAccuracy(Callback):
    def __init__(self, accuracy_threshold=.99):
        super(StopAtAccuracy, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        min_loss = logs.get('loss')
        if min_loss is not None and min_loss <= self.accuracy_threshold:
            print(f"\nReached {self.accuracy_threshold*100}% Min Loss, stopping training!")
            self.model.stop_training = True


class TemperatureScalingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_temp=1.0, **kwargs):
        super(TemperatureScalingLayer, self).__init__(**kwargs)
        self.temperature = tf.Variable(initial_temp, trainable=True, dtype=tf.float32)
        self.initial_temp = initial_temp

    def call(self, logits):
        return logits / self.temperature

    def get_config(self):
        # Base config
        config = super().get_config()
        # Add custom argument
        config.update({
            "initial_temp": self.initial_temp
        })
        return config


class MDNLayer(tf.keras.layers.Layer):
    def __init__(self, num_components, num_classes, **kwargs):
        super(MDNLayer, self).__init__(**kwargs)
        self.num_components = num_components
        self.num_classes = num_classes

        self.pi_layer = Dense(num_components * num_classes, activation='softmax')  # Mixture weights
        self.mu_layer = Dense(num_components * num_classes)  # Means
        self.log_sigma_layer = Dense(num_components * num_classes)  # Log variances

    def call(self, inputs):
        # Predict the parameters for the mixture distribution
        pi = self.pi_layer(inputs)  # Mixture weights
        mu = self.mu_layer(inputs)  # Means
        log_sigma = self.log_sigma_layer(inputs)  # Log variances
        sigma = tf.exp(log_sigma)  # Ensure positive variances
        return pi, mu, sigma

    def get_config(self):
        config = super(MDNLayer, self).get_config()
        config.update({
            "num_components": self.num_components,
            "num_classes": self.num_classes,
        })
        return config
