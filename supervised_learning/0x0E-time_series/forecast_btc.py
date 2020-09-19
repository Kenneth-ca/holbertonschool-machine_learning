#!/usr/bin/env python3
"""
Script that performs forecasting
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cleaning = __import__('preprocess_data').cleaning
preprocess = __import__('preprocess_data').preprocess
# save_csv = __import__('preprocess_data').save_csv


class WindowGenerator:
    """
    Class WindowGenerator
    """

    def __init__(self, train_df, val_df, test_df,
                 input_width, label_width, shift,
                 label_columns=None):
        """
        Window generator for the forecasting
        :param input_width:
        :param label_width:
        :param shift:
        :param train_df:
        :param val_df:
        :param test_df:
        :param label_columns:
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(
                label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        """
        Split window function
        :param features: features of the window
        :return: splited window
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Function to keep together the model
        :param data: data to process
        :return: dataset created for tensorflow
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """
        Train dataset
        :return: object train
        """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """
        Val of the dataset
        :return: object val
        """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """
        Test of the dataset
        :return: object test
        """
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """
        Function to plot many parts of the model
        :param model: model to plot
        :param plot_col: column to plot
        :param max_subplots: number of subplot
        :return: No return
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Close')


def compile_and_fit(model, window, patience=2):
    """
    Function to compile the model
    :param model: model to compile
    :param window: splited window
    :param patience: patience to get of training
    :return: history of the model
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


if __name__ == '__main__':
    filename = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    df = cleaning(filename)
    train_df, val_df, test_df = preprocess(df)
    # Use of window generator
    wide_window = WindowGenerator(train_df, val_df, test_df,
                                  input_width=24,
                                  label_width=24,
                                  shift=1, label_columns=['Close'],
                                  )
    MAX_EPOCHS = 20

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    history = compile_and_fit(lstm_model, wide_window)
    val_performance = {}
    performance = {}
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

    wide_window.plot(lstm_model)
