from collections import OrderedDict
from typing import List, Tuple
import flwr as fl
import torch
from flwr.common import Metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from models.lstm_model import Config, Net,load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#模型测试函数
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


# Define metric aggregation function
#这里应该是每个客户端一个元组，其中int表示有多个样本数据，Metrics是一个字典，里面是正确率
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # print("------配置聚合指标函数:", end="")
    # print(metrics)
    # Multiply accuracy of each client by number of examples used
    auces = [m["auc"] for num_examples, m in metrics] #

    # Aggregate and return custom metric (weighted average)
    return {"auc": np.mean(auces)}  #计算出该轮次的总体正确率

#每一轮联邦学习结束后服务器端的的评估函数，返回损失值和其他的指标字典
def evaluate_fn(server_round, parameters_ndarrays,_null)->(float,Metrics):
    params_dict = zip(net.state_dict().keys(), parameters_ndarrays)  # Python内置的打包函数，按照关键词打包
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)  # 设置参数
    loss_array, result_array,auc = test(net, testloader)
    loss_mean = np.mean(loss_array)
    return loss_mean,{'auc':auc}

config = Config()#LSTM模型的配置
net = Net(config).to(DEVICE)  #服务器维护的模型
trainloader, testloader = load_data("../../.",config)  #两个数据加载器（一个训练，一个测试）
# net.load_state_dict(torch.load("./sltm_model.pth"))
# Define strategy

#评估聚合指标函数
init_model_param =  [val.cpu().numpy() for _, val in net.state_dict().items()]
strategy = fl.server.strategy.FedAvg(initial_parameters=fl.common.ndarrays_to_parameters(init_model_param),
evaluate_metrics_aggregation_fn=weighted_average,
                                     evaluate_fn=evaluate_fn)#配置聚合指标函数 weighted_average

num_rounds = 80  #联合5次

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

print("服务器评估的自定义指标：",history_res.metrics_centralized)
print("每一轮由客户端算出的自定义指标：",history_res.metrics_distributed)
#plt.plot(history_res.metrics_distributed.get("accuracy"))
losses_centralized_list = [l for index,l in history_res.losses_centralized]
losses_distributed_list = [l for index,l in history_res.losses_distributed]

auc_centralized_list =[a for index,a in history_res.metrics_centralized.get("auc")]
auc_distributed_list = [a for index,a in history_res.metrics_distributed.get("auc")]

plt.subplot(1,2,1)
plt.plot(np.arange(num_rounds),losses_centralized_list[1:],'s-',label='由服务器评估的平均损失值',linewidth=1,markersize=3)
plt.plot(np.arange(num_rounds), losses_distributed_list,'s-',label='由客户端评估的平均损失值',linewidth=1,markersize=3)
plt.xlabel("The epoch")
plt.ylabel("Loss value")
plt.title("LSTM在横向联邦学习中")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(num_rounds),auc_centralized_list[1:],'s-',label='由服务器评估的AUC',linewidth=1,markersize=3)
plt.plot(np.arange(num_rounds), auc_distributed_list,'s-',label='由客户端评估的平均AUC',linewidth=1,markersize=3)
plt.xlabel("The epoch")
plt.ylabel("AUC value")
plt.title("LSTM在横向联邦学习中")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend()
plt.show()