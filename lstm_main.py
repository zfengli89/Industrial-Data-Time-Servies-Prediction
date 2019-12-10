# -*- coding=utf-8 -*-
"""
    Desc:
    Auth: LiZhifeng
    Date: 2019/12/10
"""
from lstm.data_processor import DataLoader
import json
import os
from lstm.run import train, predict

if __name__ == '__main__':
    configs = json.load(open('config.json', 'r'))
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