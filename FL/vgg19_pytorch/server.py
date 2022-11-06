from collections import OrderedDict
from typing import List, Tuple
import torch
from models.lstm_model import Config,Net
import flwr as fl
from flwr.common import Metrics
import matplotlib.pyplot as plt
import numpy as np
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





# Define strategy
#评估聚合指标函数
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)#配置聚合指标函数 weighted_average
num_rounds = 5
# Start Flower server
history_res =fl.server.start_server(
    server_address="127.0.0.1:8089",  #服务器地址
    config=fl.server.ServerConfig(num_rounds=num_rounds), #配置服务器的配置 ,训练的轮次
    strategy=strategy,   #策略
    grpc_max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH*3
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