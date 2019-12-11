# -*- coding=utf-8 -*-
"""
    Desc:
    Auth: LiZhifeng
    Date: 2019/12/10
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """A class for loading and transforming data for the lstm_check_point model"""

    def __init__(self, filename, split, cols_X, cols_Y):
        # read csv
        dataframe = pd.read_csv(filename)
        # 数据集
        data_train_X = dataframe.get(cols_X).values
        data_train_Y = dataframe.get(cols_Y).values
        # X规范化
        scaler_X = StandardScaler().fit(data_train_X)
        data_train_X = scaler_X.transform(data_train_X)
        # Y规范化
        self.scaler_Y = StandardScaler().fit(data_train_Y)
        data_train_Y = self.scaler_Y.transform(data_train_Y)
        # 数据集，供方法使用
        i_split = int(len(dataframe) * split)
        # 自身列预测自身
        if len(cols_X) == 1 and cols_X[0] == cols_Y[0]:
            self.data_train = data_train_X[:i_split]
            self.data_test = data_train_X[i_split:]
        else:
            self.data_train = np.hstack((data_train_Y, data_train_X))[:i_split]
            self.data_test = np.hstack((data_train_Y, data_train_X))[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_test - seq_len):
            x, y = self._next_window(i, seq_len, "test")
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_train_data(self, seq_len):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, "train")
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, "train")
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, flag):
        '''Generates the next data window from the given index location i'''
        if flag == "train":
            window = self.data_train[i:i + seq_len]
        elif flag == "test":
            window = self.data_test[i:i + seq_len]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
