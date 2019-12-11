# coding=utf-8
"""
Created on 2019/12/11 11:11

@author: EwdAger
"""

from gru.data_processor import DataLoader
import json
import os
from gru.run import train, predict

if __name__ == '__main__':
    configs = json.load(open('./config/gru_config.json', 'r'))
    data = DataLoader(
        os.path.join('data', configs['data']['train_filename']),
        configs['data']['train_test_split'],
        configs['data']['columns_X'],
        configs['data']['columns_Y'],
    )
    # 模型训练
    model = train(data, configs)
    # 模型预测
    predict(data, configs, model)
