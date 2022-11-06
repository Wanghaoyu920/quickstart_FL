import numpy as np
import pandas as  pd
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose, ToTensor, Normalize,transforms
from sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import  train_test_split
from models.lstm_model import Config
#加载林火预测数据集
"""
数据加载器，这里需要注意，和机器学习的数据不同，没一个前N天的数据组成一个序列（即一条数据）
返回的数据集Dataloader应该是(批次，天数，每天的特征值个数)  ---天数就是序列长度
"""
def load_data(csv_root_dir,config:Config):
    #加载数据文件
    source_datafile = pd.read_csv(csv_root_dir+"/ForestyFireDataSet/forestfires.csv",
                                  usecols=['FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'],
                                  dtype=np.float32,index_col=None,header=0)
    # source_datafile = pd.read_csv(csv_root_dir+"/ForestyFireDataSet/Algerian_forest_fires_dataset_UPDATE_1.csv",
    #                               usecols=['Temperature', 'RH', 'Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI','Classes'],
    #                               dtype=np.float32,index_col=None,header=0)
    # source_datafile = pd.read_csv(csv_root_dir + "/dataset/2018-2019supervised.csv", sep="\s+", header=0,
    #                               usecols=['20-20时降水量', '平均2分钟风速', '平均气温', '最小相对湿度', '火灾标签'],
    #                               dtype={'区站号': np.int32, '年': np.int32, '月': np.int32, '日': np.int32,
    #                                      '20-20时降水量': np.float32,
    #                                      '平均2分钟风速': np.float32, '平均气温': np.float32,
    #                                      '最小相对湿度': np.float32, '经纬度': np.str_,
    #                                      '火灾标签': np.int32}, index_col=None)
    if __name__=='__main__':
        print(source_datafile)

    #数据转numpy数组
    data_set = source_datafile.to_numpy()
    if __name__ == '__main__':
        print(data_set)

    #数据无量纲化处理
    data_set[:,:-1] =StandardScaler().fit_transform(data_set[:,:-1])
    if __name__ == '__main__':
        print(data_set)

    #分离出数据的x和y
    x_data_set = data_set[:,:-1] #二维矩阵
    y_data_set = data_set[:,-1]  #一个向量

    ####生成数据序列，每一个序列中报告N天的数据，例如序列的长度设定为20
    #则序列[0-19,20-39,...]  [序列数,time_step,10特征]
    x_data_seq = [(x_data_set[i:i+config.time_step,:]).tolist()  for i in range(x_data_set.shape[0]-config.time_step-1)]  #最后留出一天
    #每个序列的标签 [序列数,]
    y_data_seq = [y_data_set[i+config.time_step] for i in range(x_data_set.shape[0]-config.time_step-1)]  #每个序列对应的标签

    #转为numpy数组
    x_data_seq = np.array(x_data_seq)
    y_data_seq = np.array(y_data_seq)
    if __name__ == '__main__':
        print("求得shape-x_data_seq，y_data_seq:",x_data_seq.shape,y_data_seq.shape)

    #训练测试集分割
    train_x, test_x, train_y, test_y = train_test_split(x_data_seq, y_data_seq, random_state= config.random_seed, shuffle=config.shuffle_train_data,test_size=1-config.train_data_rate)
    if __name__ == '__main__':
        print("训练集train_x，train_y的shape:",train_x.shape,train_y.shape)
        print("测试集test_x,test_y的shape:",test_x.shape,test_y.shape)

    #转为tensor类型
    tensor_train_x,tensor_train_y = torch.FloatTensor(train_x),torch.FloatTensor(train_y)
    tensor_test_x,tensor_test_y = torch.FloatTensor(test_x),torch.FloatTensor(test_y)

    #转换成torch的DataSet
    train_data_set = Data.TensorDataset(tensor_train_x, tensor_train_y)
    test_data_set = Data.TensorDataset(tensor_test_x,tensor_test_y)
    if __name__ == '__main__':
        print(train_data_set)
        print(test_data_set)

    return DataLoader(train_data_set,batch_size=config.batch_size),DataLoader(test_data_set) #返回两个数据加载器 返回的数据应该是三维的

if __name__=='__main__':
    train_loader,test_loader=load_data("./",Config(train_data_rate=0.55,time_step=20,batch_size=1))
    print("load_data返回的结果:",train_loader,test_loader)
