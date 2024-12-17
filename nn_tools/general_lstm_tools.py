import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import TYPE_CHECKING
import math
import random

if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class BufferedBatchGenerator:
    def __init__(self, process_handler, buffer_size=50, train=True, randomize=False):
        # buffer_size: number of batches to load at once, e.g. 50
        self.ph = process_handler
        self.batch_size = self.ph.lstm_model.batch_s  # should be 16
        self.buffer_size = buffer_size  # number of batches per buffer
        self.train_tf = train
        self.sample_ind_list = []
        self.n_samples = 0

        self._current_buffer_index = -1
        self._current_buffer_data = None  # will hold arrays for x_day, x_intra, y
        self._current_sample_index = 0  # global sample index across all samples

        self.set_attributes(randomize)

    def set_attributes(self, randomize=False):
        ncols = self.ph.setup_params.num_y_cols
        hot_enc = (self.ph.lstm_data.xy_train_intra if self.train_tf
                   else self.ph.lstm_data.xy_test_intra).iloc[:, -ncols:]
        self.sample_ind_list = hot_enc[hot_enc.any(axis=1)].index.tolist()

        if randomize and self.train_tf:
            random.shuffle(self.sample_ind_list)

        self.n_samples = len(self.sample_ind_list)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self):
        self._current_buffer_index = -1
        self._current_buffer_data = None
        self._current_sample_index = 0

        return self

    def __next__(self):
        if self._current_sample_index >= self.n_samples:
            raise StopIteration

        while self._current_sample_index < self.n_samples:
            # Compute the buffer index for the current sample index
            buffer_index = self._current_sample_index // (self.buffer_size * self.batch_size)

            # If buffer not loaded or changed, load it
            if buffer_index != self._current_buffer_index:
                self._load_buffer(buffer_index)
                self._current_buffer_index = buffer_index

            # Now yield batches from the current buffer
            # local position within the buffer
            start_in_buffer = (self._current_sample_index % (self.buffer_size * self.batch_size))
            end_in_buffer = min(start_in_buffer + self.batch_size, self.buffer_size * self.batch_size)

            # Extract the batch
            x_day_batch = self._current_buffer_data[0][start_in_buffer:end_in_buffer]
            x_intra_batch = self._current_buffer_data[1][start_in_buffer:end_in_buffer]
            y_batch = self._current_buffer_data[2][start_in_buffer:end_in_buffer]

            # Update global sample index
            batch_len = end_in_buffer - start_in_buffer
            self._current_sample_index += batch_len

            # Yield the batch
            yield (x_day_batch, x_intra_batch), y_batch

    def _load_buffer(self, buffer_index):
        """
        Load a specific buffer of data into memory as arrays.
        Each buffer is buffer_size*batches_of_size(batch_size).
        """
        start = buffer_index * self.buffer_size * self.batch_size
        end = start + self.buffer_size * self.batch_size
        buffer_inds = self.sample_ind_list[start:end]
        self._current_buffer_data = self._process_buffer(buffer_inds)

    def _process_buffer(self, buffer_inds):
        """
        Process a buffer of data indices and return tensor batches.
        """
        num_y_cols = self.ph.setup_params.num_y_cols
        daily_len = self.ph.lstm_model.daily_len
        intra_len = self.ph.lstm_model.intra_len
        daily_shape = self.ph.lstm_model.input_shapes[0]
        intra_shape = self.ph.lstm_model.input_shapes[1]

        xy_intraday = self.ph.lstm_data.xy_train_intra if self.train_tf else self.ph.lstm_data.xy_test_intra
        x_daily = self.ph.lstm_data.x_train_daily if self.train_tf else self.ph.lstm_data.x_test_daily

        xy_intraday = xy_intraday.to_numpy() if hasattr(xy_intraday, "to_numpy") else xy_intraday
        x_daily = x_daily.to_numpy() if hasattr(x_daily, "to_numpy") else x_daily

        x_day_buffer, x_intra_buffer, y_buffer = [], [], []

        for t_ind in buffer_inds:
            trade_dt = xy_intraday[t_ind, 0]

            x_daily_input = x_daily[x_daily[:, 0] < trade_dt][-daily_len:, 1:]
            x_intra_input = xy_intraday[t_ind - intra_len:t_ind, 1:-num_y_cols]
            y_input = xy_intraday[t_ind, -num_y_cols:]

            x_day_buffer.append(x_daily_input)
            x_intra_buffer.append(x_intra_input)
            y_buffer.append(y_input)

        x_day_tensor = np.array(x_day_buffer, dtype=np.float32).reshape(len(buffer_inds), daily_len, daily_shape[1])
        x_intra_tensor = np.array(x_intra_buffer, dtype=np.float32).reshape(len(buffer_inds), intra_len, intra_shape[1])
        y_tensor = np.array(y_buffer, dtype=np.float32)

        return x_day_tensor, x_intra_tensor, y_tensor

    def generate_tf_dataset(self):
        # Wrap the Python generator as a TF Dataset.
        daily_shape = self.ph.lstm_model.input_shapes[0]
        intra_shape = self.ph.lstm_model.input_shapes[1]

        # from_generator requires a callable that returns an iterator
        def gen():
            for (xd, xi), y in self:
                yield (xd, xi), y

        t_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, daily_shape[0], daily_shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, intra_shape[0], intra_shape[1]), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(None, self.ph.setup_params.num_y_cols), dtype=tf.float32),
            )
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return t_dataset

    def load_full_dataset(self):
        """
        Load the entire dataset into memory and return it as tensors.
        """
        x_day_full, x_intra_full, y_full = self._process_buffer(self.sample_ind_list)

        full_dataset = tf.data.Dataset.from_tensor_slices(((x_day_full, x_intra_full),
                                                           y_full)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return full_dataset


class BufferedBatchSequence(Sequence):
    def __init__(self, process_handler, buffer_size=50, train=True, randomize=False):
        self.generator = BufferedBatchGenerator(process_handler, buffer_size, train, randomize)

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        return next(self.generator)

    def on_epoch_end(self):
        # Shuffle at the end of each epoch if needed
        self.generator.set_attributes(randomize=True)


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


