import warnings
from collections import OrderedDict
import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
from models import vgg19_model

warnings.filterwarnings("ignore", category=UserWarning)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE  = "cuda"

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#训练函数
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()  #损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #优化器
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):  #tqdm用于显示进度条
            optimizer.zero_grad()  #清除优化器中的梯度
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()  #反向传播，更新梯度
            optimizer.step()  #梯度下降，模型更新

#测试函数
def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

#数据加载
def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #正则化，然后ToTensor
    trainset = CIFAR10("../../data", train=True, download=True, transform=trf) #训练集
    testset = CIFAR10("../../data", train=False, download=True, transform=trf) #测试集
    return DataLoader(trainset, batch_size=256, shuffle=True), DataLoader(testset,batch_size=256)  #返回两个数据加载器


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# net = Net().to(DEVICE)  #生成卷积神经网络模型
net = vgg19_model.get_model().to(DEVICE)
trainloader, testloader = load_data()  #两个数据加载器（一个训练，一个测试）
print("模型大小为:",sys.getsizeof(net),"Byte")
# Define Flower client
#派生一个客户端类，父类为使用 NumPy 的 Flower 客户端的抽象基类。
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        param =  [val.cpu().numpy() for _, val in net.state_dict().items()]
        print("本地模型的参数大小为（Byte）:",sys.getsizeof(param))
        return param

    def set_parameters(self, parameters):
        print("设置接收到的服务器参数大小（Byte）:",sys.getsizeof(parameters))
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # print("我在训练:",parameters,config)
        #从服务器接收的（全局）模型参数和用于自定义本地训练过程的配置值字典的训练指令。
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}  #有三个返回值

    def evaluate(self, parameters, config):
        # print("我在进行评估:",parameters, config)
        # 服务器接收的（全局）模型参数和用于自定义本地评估过程的配置值字典。
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8089",
    client=FlowerClient(), #将自定义的客户端类作为客户端
    grpc_max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH*3
)
