import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.lstm_model import Net,Config
from ForestyFireDataSet.dataload_forlstm import load_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
这是一个序列预测问题，例如给定序列 1 2 3 4 5  预测下一个是6
因此对于该问题而言，没天的各项数据就是一个序列元素，我们需要根据N天的数据，来预测下一天的元素值。

"""


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

#模型训练函数
def train(net,config,train_loader,test_loader):

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate) #优化器
    criterion = torch.nn.CrossEntropyLoss()      # 这两句是定义优化器和loss
    valid_loss_min = float("inf")  #正无穷
    bad_epoch = 0
    for epoch in range(config.epoch):
        net.train()                   # pytorch中，训练时要转换成训练模式
        train_loss_array = []  #存储训练过程中的损失值
        hidden_train = None
        for i, _data in enumerate(train_loader): #enumerate使得在原有的迭代对象前面出现一个下标i
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
            #print("train_loss:",loss)

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        valid_loss_array,valid_result_array,acc = test(net,test_loader)  #评估当前的模型,得到目前的损失值数组,预测结果数组和正确率
        train_loss_cur_sum = np.sum(train_loss_array)  #训练loss的和值
        valid_loss_cur_sum = np.sum(valid_loss_array)  #评估loss的和值
        print(f"epoch:{epoch}/{config.epoch}  train_loss_sum:{train_loss_cur_sum} valid_loss_sum:{valid_loss_cur_sum} "
              f"accuracy:{acc}")
        if valid_loss_cur_sum < valid_loss_min:  #存储最好的模型，如果当前的平均loss小于历史最小的loss
            valid_loss_min = valid_loss_cur_sum
            bad_epoch = 0
            torch.save(net.state_dict(), "./lstm_model.pth")  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                print(" The training stops early in epoch {}".format(epoch))
                break



def draw_res(valid_result_array,test_loader):
    y_label = []
    for _,l in test_loader:
        y_label.append(int(l.item()))
    print(y_label)
    print(valid_result_array)
    plt.plot(np.arange(len(y_label)),y_label,color='red',label='实际值')
    plt.plot(np.arange(len(y_label)),valid_result_array,color='green',label='预测值')
    plt.xlabel("Time")
    plt.ylabel("Is fire")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend()
    plt.show()
    # valid_result_array = np.array(valid_result_array)
    # y_label = np.array(y_label)



if __name__=="__main__":
    config = Config()
    print("Please wait...")
    np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
    net = Net(config)  #创建网络
    net.load_state_dict(torch.load("./lstm_model.pth"))
    net = net.to(DEVICE)
    print(".....Net has created...")
    train_loader,test_loader = load_data("../",config)  #加载数据集
    print(".....start train....")
    train(net,config,train_loader,test_loader) #训练模型

    _,valid_result_array,acc=test(net,test_loader)  #测试
    draw_res(valid_result_array,test_loader)

