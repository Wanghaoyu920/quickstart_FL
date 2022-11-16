import warnings
from collections import OrderedDict
import sys
import flwr as fl
import numpy as np
import torch
from tqdm import tqdm
from models.lstm_model import Net,Config
from history_data_dir.ForestyFireDataSet.dataload_forlstm import load_data
warnings.filterwarnings("ignore", category=UserWarning)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#模型训练函数
def train(net,config,train_loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate) #优化器
    criterion = torch.nn.CrossEntropyLoss()      # 这两句是定义优化器和loss
    # for epoch in range(config.epoch):
    net.train()                   # pytorch中，训练时要转换成训练模式
    train_loss_array = []  #存储训练过程中的损失值
    hidden_train = None
    for i, _data in enumerate(tqdm(train_loader)): #enumerate使得在原有的迭代对象前面出现一个下标i
        _train_X, _train_Y = _data[0].to(DEVICE),_data[1].to(DEVICE)
        optimizer.zero_grad()               # 训练前要将梯度信息置 0
        pred_Y, hidden_train = net(_train_X, hidden_train)    # 这里走的就是前向计算forward函数
        hidden_train = None
        # h_0, c_0 = hidden_train
        # h_0.detach_(), c_0.detach_()    # 去掉hidden的梯度信息
        # hidden_train = (h_0, c_0)
        # print(pred_Y, _train_Y)
        loss = criterion(pred_Y, _train_Y.long())  # 计算loss
        loss.backward()                     # 将loss反向传播
        optimizer.step()                    # 用优化器更新参数
        train_loss_array.append(loss.item())  #保存训练过程中的损失值
        # print("train_loss:",loss)
    train_loss_cur_sum = np.sum(train_loss_array)  #训练loss的和值
    print(f"train_loss_sum:{train_loss_cur_sum} ")

#模型测试函数
def test(net,test_loader):
    net.eval()  # pytorch中，预测时要转换成预测模式
    criterion = torch.nn.CrossEntropyLoss()  # 定义loss
    valid_loss_array = []  #存储评估的损失值
    valid_result_array = []  #存储评估的结果值
    valid_y_array = []  #存储正确的标签结果
    hidden_valid = None
    for _valid_X, _valid_Y in test_loader:
        valid_y_array.append(int(_valid_Y.item()))
        _valid_X, _valid_Y = _valid_X.to(DEVICE), _valid_Y.to(DEVICE)
        pred_Y, hidden_valid = net(_valid_X, hidden_valid)
        loss = criterion(pred_Y, _valid_Y.long())  # 验证过程只有前向计算，无反向传播过程
        valid_loss_array.append(loss.item())
        pred_Y = pred_Y.cpu()
        # print(pred_Y)
        # print(torch.argmax(pred_Y))
        valid_result_array.append(torch.argmax(pred_Y).item())
    valid_result_array = np.array(valid_result_array)
    valid_y_array = np.array(valid_y_array)
    acc = len(valid_y_array[valid_result_array==valid_y_array])/len(valid_result_array)
    return valid_loss_array,valid_result_array,acc

config = Config() #自定义LSTM网络的相关配置类
net = Net(config).to(DEVICE)  #创建LSTM网络
print("模型大小为:",sys.getsizeof(net),"Byte")
trainloader, testloader = load_data("../../",config)  #两个数据加载器（一个训练，一个测试）

# Define Flower client
#派生一个联邦学习的客户端类，父类为使用 NumPy 的 Flower 客户端的抽象基类。
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, fl_config):
        param =  [val.cpu().numpy() for _, val in net.state_dict().items()]
        print("本地模型的参数大小为（Byte）:",sys.getsizeof(param))
        return param

    def set_parameters(self, parameters):
        print("接收到服务器的参数大小（Byte）:",sys.getsizeof(parameters))
        params_dict = zip(net.state_dict().keys(), parameters)  #Python内置的打包函数，按照关键词打包
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)  #设置从服务器接收到的参数

    def fit(self, parameters, fl_config):
        print("开启本地训练,接收到服务器的联邦学习配置为:",fl_config)
        #从服务器接收的（全局）模型参数和用于自定义本地训练过程的配置值字典的训练指令。
        self.set_parameters(parameters)  #更新本地模型参数
        train(net, config,trainloader)  #训练
        print("本地训练结束，训练的数据条数:",len(trainloader.dataset))
        return self.get_parameters(fl_config={}), len(trainloader.dataset), {}  #有三个返回值

    def evaluate(self, parameters, fl_config):
        print("开启本地评估,接收到服务器的联邦学习配置为:",fl_config)
        # 服务器接收的（全局）模型参数和用于自定义本地评估过程的配置值字典。
        self.set_parameters(parameters)
        loss_array,result_array, accuracy = test(net,testloader)
        loss_sum = sum(loss_array)
        print(f"本地评估结束,loss_sum:{loss_sum}, accuracy:{accuracy}, 测试数据数:{len(testloader.dataset)}")
        return sum(loss_array), len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8091",
    client=FlowerClient(), #将自定义的客户端类作为客户端
    grpc_max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH
)
