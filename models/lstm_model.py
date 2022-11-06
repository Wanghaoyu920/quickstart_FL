# -*- coding: UTF-8 -*-
"""
@author: hichenway
@知乎: 海晨威
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
pytorch 模型
"""

import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Config:

    def __init__(self,train_data_rate=None,batch_size=None,time_step=None):
        if batch_size!=None:
            self.batch_size = batch_size
        if train_data_rate != None:
            self.train_data_rate = train_data_rate
        if time_step != None:
            self.time_step = time_step

    #网络参数
    input_size = 8             #LSTM 模型的输入层大小，即模型的参数
    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小，也是Liner的输入层大小
    output_size = 2             #Liner层的输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.3          # dropout概率
    time_step = 36              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    do_train = True
    do_predict = True
    #add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle

    train_data_rate = 0.45      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate

    batch_size = 1
    learning_rate = 0.0002
    epoch = 200                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 3                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

class Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)


    def forward(self, x, hidden=None):
        lstm_out_, hidden = self.lstm(x, hidden)  #这个hidden就是保证每次循环的时候能够融入之前的数据
        lstm_out = lstm_out_[:,-1,:]  #shape：（batch，seq_len，input_size）。这个lstm_out 就是最后一天的数据传进LSTM 后，得到的输出结果。
        # 也就是我们把整个序列的所有行都穿进去以后，LSTM 给出的结果。
        linear_out = self.linear(lstm_out)
        # linear_out = linear_out.squeeze(-1) #以达到降低一维的目的~
        #linear_out = self.sigmoid(linear_out)
        return linear_out, hidden


