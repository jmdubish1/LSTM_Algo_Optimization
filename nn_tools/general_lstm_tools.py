import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import TYPE_CHECKING
import math
import random

if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class BufferedBatchGenerator(Sequence):
    def __init__(self, process_handler, buffer_size=50, train=True, randomize=False):
        self.ph = process_handler
        self.batch_size = self.ph.lstm_model.batch_s
        self.buffer_size = buffer_size
        self.train_tf = train
        self.sample_ind_list = []
        self.n_samples = 0
        self._current_index = 0
        self.set_attributes(randomize)

        self.xy_intraday = (self.ph.lstm_data.xy_train_intra if self.train_tf
                            else self.ph.lstm_data.xy_test_intra).to_numpy()
        self.x_daily = (self.ph.lstm_data.x_train_daily if self.train_tf
                        else self.ph.lstm_data.x_test_daily).to_numpy()

    def set_attributes(self, randomize=False):
        ncols = self.ph.setup_params.num_y_cols
        hot_enc = (self.ph.lstm_data.xy_train_intra if self.train_tf
                   else self.ph.lstm_data.xy_test_intra).iloc[:, -ncols:]
        self.sample_ind_list = hot_enc[hot_enc.any(axis=1)].index.tolist()

        if randomize and self.train_tf:
            random.shuffle(self.sample_ind_list)

        self.n_samples = len(self.sample_ind_list)

    def __iter__(self):
        self._current_index = 0  # Reset batch index when iteration starts
        return self

    def __next__(self):
        if self._current_index >= self.__len__():
            raise StopIteration  # End of iteration

        # Retrieve batch at the current index
        batch = self.__getitem__(self._current_index)
        self._current_index += 1
        return batch

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        # Batch slicing logic
        start = index * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        buffer_inds = self.sample_ind_list[start:end]
        return self._process_buffer(buffer_inds)

    def on_epoch_end(self):
        """Shuffle the data at the end of each epoch if training."""
        if self.train_tf:
            random.shuffle(self.sample_ind_list)

    def _process_buffer(self, buffer_inds):
        num_y_cols = self.ph.setup_params.num_y_cols
        daily_len = self.ph.lstm_model.daily_len
        intra_len = self.ph.lstm_model.intra_len
        # print(self.ph.lstm_model.model.input_shape)
        # breakpoint()
        daily_shape = self.ph.lstm_model.model.input_shape[0]
        intra_shape = self.ph.lstm_model.model.input_shape[1]

        x_day_buffer, x_intra_buffer, y_buffer = [], [], []

        for t_ind in buffer_inds:
            trade_dt = self.xy_intraday[t_ind, 0]

            x_daily_input = self.x_daily[self.x_daily[:, 0] < trade_dt][-daily_len:, 1:]
            x_intra_input = self.xy_intraday[t_ind - intra_len:t_ind, 1:-num_y_cols]
            y_input = self.xy_intraday[t_ind, -num_y_cols:]

            x_day_buffer.append(x_daily_input)
            x_intra_buffer.append(x_intra_input)
            y_buffer.append(y_input)

        x_day_tensor = tf.convert_to_tensor(x_day_buffer, dtype=tf.float32)
        x_intra_tensor = tf.convert_to_tensor(x_intra_buffer, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_buffer, dtype=tf.float32)

        # Reshape tensors to match expected shapes
        x_day_tensor = tf.reshape(x_day_tensor, [len(buffer_inds), daily_len, daily_shape[2]])
        x_intra_tensor = tf.reshape(x_intra_tensor, [len(buffer_inds), intra_len, intra_shape[2]])

        return (x_day_tensor, x_intra_tensor), y_tensor

    def generate_tf_dataset(self):
        # Wrap the Python generator as a TF Dataset.
        daily_shape = self.ph.lstm_model.input_shapes[0]
        intra_shape = self.ph.lstm_model.input_shapes[1]

        t_dataset = tf.data.Dataset.from_generator(
            lambda: iter(self),
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, daily_shape[0], daily_shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, intra_shape[0], intra_shape[1]), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(None, self.ph.setup_params.num_y_cols), dtype=tf.float32),
            )
        ).prefetch(tf.data.AUTOTUNE)

        return t_dataset

    def load_full_dataset(self):
        """
        Load the entire dataset into memory and return it as tensors.
        """
        (x_day_full, x_intra_full), y_full = self._process_buffer(self.sample_ind_list)

        full_dataset = tf.data.Dataset.from_tensor_slices(((x_day_full, x_intra_full),
                                                           y_full)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return full_dataset


def one_cycle_lr(initial_lr, total_epochs):
    """
    Returns a callable One Cycle Learning Rate Scheduler.

    Parameters:
    - initial_lr (float): Peak learning rate.
    - total_epochs (int): Total number of epochs.

    Returns:
    - A function that computes the learning rate for each epoch.
    """
    max_lr = initial_lr  # Peak learning rate
    min_lr = .0000005  # Minimum learning rate

    def schedule(epoch, lr):
        if epoch < total_epochs // 2:
            # Increase learning rate linearly to the peak
            return min_lr + (max_lr - min_lr) * (epoch / (total_epochs // 2))
        else:
            # Decrease learning rate following a cosine decay
            return min_lr + (max_lr - min_lr) * \
                   (1 + math.cos(math.pi * (epoch - total_epochs // 2) / (total_epochs // 2))) / 2

    return schedule


