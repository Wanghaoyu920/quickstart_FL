from collections import OrderedDict
from typing import List, Tuple
import flwr as fl
import torch
from flwr.common import Metrics
import matplotlib.pyplot as plt
import numpy as np
from ForestyFireDataSet.dataload_forlstm import load_data
from models.lstm_model import Config, Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Define metric aggregation function
#这里应该是每个客户端一个元组，其中int表示有多个样本数据，Metrics是一个字典，里面是正确率
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # print("------配置聚合指标函数:", end="")
    # print(metrics)
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] #求解每个客户端中预测正确的个数
    examples = [num_examples for num_examples, _ in metrics] #求解每个客户端中的样本数
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}  #计算出该轮次的总体正确率

#每一轮联邦学习结束后服务器端的的评估函数，返回损失值和其他的指标字典
def evaluate_fn(server_round, parameters_ndarrays,_null)->(float,Metrics):
    params_dict = zip(net.state_dict().keys(), parameters_ndarrays)  # Python内置的打包函数，按照关键词打包
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)  # 设置参数
    loss_array, result_array, accuracy = test(net, testloader)
    loss_sum = sum(loss_array)
    return loss_sum,{"accuracy": accuracy}

config = Config()#LSTM模型的配置
net = Net(config).to(DEVICE)  #服务器维护的模型
trainloader, testloader = load_data("../../",config)  #两个数据加载器（一个训练，一个测试）
# net.load_state_dict(torch.load("./sltm_model.pth"))
# Define strategy

#评估聚合指标函数
init_model_param =  [val.cpu().numpy() for _, val in net.state_dict().items()]
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,initial_parameters=fl.common.ndarrays_to_parameters(init_model_param),
                                     evaluate_fn=evaluate_fn)#配置聚合指标函数 weighted_average

num_rounds = 5  #联合5次

# Start Flower server
history_res =fl.server.start_server(
    server_address="127.0.0.1:8091",  #服务器地址
    config=fl.server.ServerConfig(num_rounds=num_rounds), #配置服务器的配置 ,训练的轮次
    strategy=strategy,   #策略
    grpc_max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH
)

#返回本次联邦学习的训练过程指标
print("服务器评估的损失值：",history_res.losses_centralized)
print("每一轮由客户端评估的损失值：",history_res.losses_distributed)

print("服务器评估的损失值：",history_res.metrics_centralized)
print("每一轮由客户端算出的准确度：",history_res.metrics_distributed)
#plt.plot(history_res.metrics_distributed.get("accuracy"))
acc_list= history_res.metrics_distributed.get('accuracy')
acc=[acc_ for _,acc_  in acc_list]
plt.plot(np.arange(num_rounds),acc)
plt.show()