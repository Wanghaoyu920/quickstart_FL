
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.utils.data as Data
import  pandas as pd

class Config:

    def __init__(self,train_data_rate=None,batch_size=None,time_step=None):
        if batch_size!=None:
            self.batch_size = batch_size
        if train_data_rate != None:
            self.train_data_rate = train_data_rate
        if time_step != None:
            self.time_step = time_step


    #网络参数
    input_size = 12             #LSTM 模型的输入层大小，即模型的参数
    hidden_size = 64           # LSTM的隐藏层大小，也是输出大小，也是Liner的输入层大小
    output_size = 1             #Liner层的输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.5          # dropout概率
    time_step = 36              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    do_train = True
    do_predict = True
    #add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle

    train_data_rate = 0.65      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    dataset_days = 366           #数据集有多少天
    batch_size = 32
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
        linear_out = linear_out.squeeze(-1) #以达到降低一维的目的~
        #linear_out = self.sigmoid(linear_out)
        return linear_out, hidden


#加载数据
def load_data(csv_root_dir,config:Config):
    # 加载数据文件
    source_datafile = pd.read_csv(csv_root_dir + "/data/崇礼区2016数据整合导出_hy_OK.csv",
                                  usecols=['坡度均值','坡向众数','EVI植被指数','到公路距离','到河道距离','PRS','TEM','RHU','PRE','WIN','SSD','GST','Fire'],
                                  dtype=np.float32, index_col=None, header=0)

    if __name__=='__main__':
        print(source_datafile)

    # 数据转numpy数组
    data_set = source_datafile.to_numpy()
    if __name__ == '__main__':
        print(data_set)

    # 数据无量纲化处理
    data_set[:, :-1] = StandardScaler().fit_transform(data_set[:, :-1])
    if __name__ == '__main__':
        print(data_set)

    #分离出数据的x和y
    x_data_set = data_set[:,:-1] #二维矩阵
    y_data_set = data_set[:,-1]  #一个向量

    ####生成数据序列，每一个序列中报告N天的数据，例如序列的长度设定为20
    # 则序列[0-19,20-39,...]  [序列数,time_step,10特征]
    x_data_seq = []
    for fid in  range(2510):  #每一个fid代表一个小方格，即地理上的1km*1km区域。将崇礼区划分为2510个区域。
        x_data_seq.extend(
            [ (x_data_set[fid*config.dataset_days + i: fid*config.dataset_days + i + config.time_step, :]).tolist() for i in range(config.dataset_days - config.time_step - 1)]
                    )
    # 每个序列的标签 [序列数,]
    y_data_seq = []
    for fid in range(2510):
        y_data_seq.extend(
            [ (y_data_set[fid*config.dataset_days + i +config.time_step]) for i in range(config.dataset_days - config.time_step -1)]
        )
    x_data_seq = np.array(x_data_seq)
    y_data_seq = np.array(y_data_seq)
    if __name__ == '__main__':
        print("求得shape-x_data_seq，y_data_seq:", x_data_seq.shape, y_data_seq.shape)

        # 训练测试集分割
    train_x, test_x, train_y, test_y = train_test_split(x_data_seq, y_data_seq, random_state=config.random_seed,
                                                        shuffle=config.shuffle_train_data,
                                                        test_size=1 - config.train_data_rate)
    if __name__ == '__main__':
        print("训练集train_x，train_y的shape:", train_x.shape, train_y.shape)
        print("测试集test_x,test_y的shape:", test_x.shape, test_y.shape)

    # 转为tensor类型
    tensor_train_x, tensor_train_y = torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    tensor_test_x, tensor_test_y = torch.FloatTensor(test_x), torch.FloatTensor(test_y)

    # 转换成torch的DataSet
    train_data_set = Data.TensorDataset(tensor_train_x, tensor_train_y)
    test_data_set = Data.TensorDataset(tensor_test_x, tensor_test_y)
    if __name__ == '__main__':
        print(train_data_set)
        print(test_data_set)

    return DataLoader(train_data_set, batch_size=config.batch_size), DataLoader(test_data_set,
                                                                                batch_size=config.batch_size)  # 返回两个数据加载器 返回的数据应该是三维的
if __name__=='__main__':
    config= Config()
    train_loader,test_loader=load_data("../.", config)
    print("load_data返回的结果:",train_loader,test_loader)