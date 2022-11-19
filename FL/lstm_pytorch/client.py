import warnings
from collections import OrderedDict
import sys
import flwr as fl
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from models.lstm_model import Net,Config,load_data
warnings.filterwarnings("ignore", category=UserWarning)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#模型测试函数
def test(net,test_loader):
    criterion = torch.nn.MSELoss()  # 定义loss
    valid_loss_array = []  # 存储评估的损失值
    valid_result_array = []  # 存储评估的结果值
    valid_y_array = []  # 存储正确的标签结果
    hidden_valid = None
    for _valid_X, _valid_Y in tqdm(test_loader):
        valid_y_array.extend([(vv.item()) for vv in _valid_Y])
        _valid_X, _valid_Y = _valid_X.to(DEVICE), _valid_Y.to(DEVICE)
        pred_Y, hidden_valid = net(_valid_X, hidden_valid)
        hidden_valid = None
        loss = criterion(pred_Y, _valid_Y.float())  # 验证过程只有前向计算，无反向传播过程
        valid_loss_array.append(loss.item())
        pred_Y = pred_Y.cpu()
        # print(pred_Y)
        # print(torch.argmax(pred_Y))
        # valid_result_array.append(torch.argmax(pred_Y).item())
        valid_result_array.extend([pp.item() for pp in pred_Y])
    valid_result_array = np.array(valid_result_array)
    valid_y_array = np.array(valid_y_array)
    valid_result_array[valid_result_array > 1.0] = 1.0
    valid_result_array[valid_result_array < 0.0] = 0.0
    auc = roc_auc_score(valid_y_array, valid_result_array)
    # acc = len(valid_y_array[valid_result_array==valid_y_array])/len(valid_result_array)
    return valid_loss_array, valid_result_array, auc



#模型训练函数
def train(net,config,train_loader,test_loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)  # 优化器
    # criterion = torch.nn.CrossEntropyLoss()      # 这两句是定义优化器和loss
    criterion = torch.nn.MSELoss()  # 这两句是定义优化器和loss

    epochs_train_loss = []  # 存储训练过程中的损失值
    epochs_test_loss = []  # 存储测试损失值
    epochs_test_auc = []  # 存储测试auc

    valid_loss_min = float("inf")  # 正无穷
    bad_epoch = 0

    for epoch in range(config.epoch):
        net.train()  # pytorch中，训练时要转换成训练模式
        train_loss_array = []  # 存储训练过程中的损失值
        hidden_train = None
        print(f"\n***training epoch:{epoch + 1}/{config.epoch} ")
        for i, _data in enumerate(tqdm(train_loader)):  # enumerate使得在原有的迭代对象前面出现一个下标i
            _train_X, _train_Y = _data[0].to(DEVICE), _data[1].to(DEVICE)
            optimizer.zero_grad()  # 训练前要将梯度信息置 0
            pred_Y, hidden_train = net(_train_X, hidden_train)  # 这里走的就是前向计算forward函数
            hidden_train = None
            # h_0, c_0 = hidden_train
            # h_0.detach_(), c_0.detach_()    # 去掉hidden的梯度信息
            # hidden_train = (h_0, c_0)
            # print(pred_Y, _train_Y)
            loss = criterion(pred_Y, _train_Y.float())  # 计算loss
            loss.backward()  # 将loss反向传播
            optimizer.step()  # 用优化器更新参数
            train_loss_array.append(loss.item())  # 保存训练过程中的损失值
            # print("train_loss:",loss)
        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        valid_loss_array, valid_result_array,auc = test(net, test_loader)  # 评估当前的模型,得到目前的损失值数组,预测结果数组和正确率
        train_loss_cur_mean = np.mean(train_loss_array)  # 训练loss的均值
        valid_loss_cur_mean = np.mean(valid_loss_array)  # 评估loss的均值

        print(f"train_loss_mean:{train_loss_cur_mean} valid_loss_mean:{valid_loss_cur_mean} \n")
        epochs_train_loss.append(train_loss_cur_mean)
        epochs_test_loss.append(valid_loss_cur_mean)
        epochs_test_auc.append(auc)
    return epochs_train_loss,epochs_test_loss,epochs_test_auc


config = Config() #自定义LSTM网络的相关配置类
net = Net(config).to(DEVICE)  #创建LSTM网络
print("模型大小为:",sys.getsizeof(net),"Byte")
trainloader, testloader = load_data("../../.",config)  #两个数据加载器（一个训练，一个测试）

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
        config.epoch=1 #修改训练次数
        train(net, config,trainloader,testloader)  #训练
        print("本地训练结束，训练的数据条数:",len(trainloader.dataset))
        return self.get_parameters(fl_config={}), len(trainloader.dataset), {}  #有三个返回值

    def evaluate(self, parameters, fl_config):
        print("开启本地评估,接收到服务器的联邦学习配置为:",fl_config)
        # 服务器接收的（全局）模型参数和用于自定义本地评估过程的配置值字典。
        self.set_parameters(parameters)
        loss_array,result_array,auc = test(net,testloader)
        print(f"本地评估结束,loss_mean:{np.mean(loss_array)}, 测试数据数:{len(testloader.dataset)}")
        return np.mean(loss_array), len(testloader.dataset), {'auc':auc}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8091",
    client=FlowerClient(), #将自定义的客户端类作为客户端
    grpc_max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH
)
