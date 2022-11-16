import numpy as np
import pandas as  pd
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose, ToTensor, Normalize,transforms
from sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import  train_test_split

#加载林火预测数据集
def load_data(csv_root_dir,test_rate,batch_size):
    source_datafile = pd.read_csv(csv_root_dir+"/2018-2019supervised.csv",sep="\s+",header=0,
                                  usecols=['20-20时降水量','平均2分钟风速','平均气温','最小相对湿度','火灾标签'],
                                  dtype={'区站号':np.int32,'年':np.int32,'月':np.int32,'日':np.int32,'20-20时降水量':np.float32,
                                         '平均2分钟风速':np.float32,'平均气温':np.float32,'最小相对湿度':np.float32,'经纬度':np.str_,
                                         '火灾标签':np.int32},index_col=None)
    if __name__=='__main__':
        print(source_datafile)
    data_set = source_datafile.to_numpy()
    if __name__ == '__main__':
        print(data_set)

    #数据无量纲化处理
    data_set[:,:-1] =StandardScaler().fit_transform(data_set[:,:-1])

    if __name__ == '__main__':
        print(data_set)

    train_x, test_x, train_y, test_y = train_test_split(data_set[:, :-1], data_set[:, -1], shuffle=False,test_size=test_rate)
    if __name__ == '__main__':
        print(train_y,test_y)
        print(train_x.shape,test_x.shape)

    tensor_train_x,tensor_train_y = torch.FloatTensor(train_x),torch.FloatTensor(train_y)
    tensor_test_x,tensor_test_y = torch.FloatTensor(test_x),torch.FloatTensor(test_y)
    # print(tensor_train_x.shape,tensor_train_y.shape)
    #转换成torch的DataLoader
    train_data_set = Data.TensorDataset(tensor_train_x, tensor_train_y)
    test_data_set = Data.TensorDataset(tensor_test_x,tensor_test_y)
    if __name__ == '__main__':
        print(train_data_set)
        print(test_data_set)

    return DataLoader(train_data_set,batch_size=batch_size),DataLoader(test_data_set) #返回两个数据加载器

if __name__=='__main__':
    train_loader,test_loader=load_data("/", 0.3, 64)
