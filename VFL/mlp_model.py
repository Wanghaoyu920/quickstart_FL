import torch
import torch.nn as nn


#现在这个配置是多层感知机的
class Config:

    def __init__(self,train_data_rate=None,batch_size=None,time_step=None):
        if batch_size!=None:
            self.batch_size = batch_size
        if train_data_rate != None:
            self.train_data_rate = train_data_rate
        if time_step != None:
            self.time_step = time_step

    #网络参数
    input_size = 13             #LSTM 模型的输入层大小，即模型的参数
    hidden_size = 64           # LSTM的隐藏层大小，也是输出大小，也是Liner的输入层大小
    output_size = 1             #Liner层的输出大小
    # lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.5          # dropout概率
    time_step = 31              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它
    #time_step不能乱改，依赖于提供的数据集文件

    good_value = 0.548  #模型输出的阈值

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle

    train_data_rate = 0.70      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    dataset_days = 366           #数据集有多少天
    squares = 2001          #数据集中有多少个方格

    batch_size = 32  #目前的crypten只支持1的情况，因为要求模型输出的结果是1
    learning_rate = 0.001
    epoch = 420                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 10                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

class MLP(nn.Module):
    def __init__(self, config:Config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear((config.time_step-1)*config.input_size, 1024)
        self.fc2 = nn.Linear(1024, config.output_size)

    def forward(self, x: torch.Tensor):
        res = self.fc1(x)
        res = torch.relu(res)
        # res = torch.sigmoid(res)
        # res = torch.tanh(res)
        res = self.fc2(res)
        res = res.view(-1) #转换为1维
        return res
