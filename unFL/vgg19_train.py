import warnings

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import vgg19_model
from history_data_dir.dataset.dataload_forvgg import load_data
warnings.filterwarnings("ignore", category=UserWarning)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE  = "cuda"

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
    # return loss / len(testloader.dataset), correct / total
    return loss , correct / total


net = vgg19_model.get_model().to(DEVICE)
trainloader, testloader = load_data("../history_data_dir/dataset")  #两个数据加载器（一个训练，一个测试）

num_rounds = 100 #训练轮次
writer = SummaryWriter('../logs/unFl_vgg19') #使用tensorboard可视化
print("非联邦学习的训练，当前设置的训练轮次为"+str(num_rounds))
print("训练开始...")

for round in range(num_rounds):
    train(net,trainloader,1)
    loss, acc = test(net,testloader)
    print(f"第{round+1}轮训练结束,loss:{loss},acc:{acc}.....")
    writer.add_scalar("loss",loss,round+1)
    writer.add_scalar("acc",acc,round+1)
print("训练结束...")