# coding=utf-8
"""
Created on 2019/12/11 11:04

@author: EwdAger
"""

import os, math, pickle
from gru.model import Model
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def train(data, configs):
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    model = Model()
    model.build_model(configs)
    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir'],
        logs_dir=configs['model']['logs_dir']
    )
    # 模型保存
    if not os.path.exists(configs["model"]["model_dir"]):
        os.makedirs(configs["model"]["model_dir"])
    pickle.dump(model, open(
        os.path.join(configs["model"]["model_dir"], "model_" + dt.datetime.now().strftime('%d%m%Y-%H%M%S')), "wb"))
    return model


def predict(data, configs, model_file):
    if isinstance(model_file, Model):
        model = model_file
    else:
        model = pickle.load(open(os.path.join(configs["model"]["model_dir"], model_file), "rb"))
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length']
    )
    predictions = model.predict_point_by_point(x_test)
    plot_results(predictions, y_test)
    print("测试集的r2 score为：{}".format(r2_score(y_test, predictions)))


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def evaluate():
    pass