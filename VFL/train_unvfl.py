import random
import sys
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from mlp_model import MLP,Config
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = ["气象", "地理加火灾标签"]
feature_sizes = [8,6]

def load_csv_numpy(config:Config)-> np.ndarray:

    source_datafile = pd.read_csv("./data/田林县2016数据集_训练集数据_序列30_OK.csv",
                                      usecols=['坡度','坡向','EVI','到河道距离','到公路距离','PRS','TEM','RHU','PRE','WIN','WIN_Dir','SSD','GST','Fire'],
                                      dtype=np.float32, index_col=None, header=0)
    data_set = source_datafile.to_numpy()
    # print(f"源文件shape：{data_set.shape}")
    return data_set
#生成dataloader
def make_dataloader(config:Config):
    # mpc_tensor = load_encrypt_tensor(feature_name, config)  #将数据加载到cryptensor中
    normal_tensor = load_csv_numpy(config)  #将数据加载到numpy中

    x_data_set, y_data_set = normal_tensor[:, :-1], normal_tensor[:, -1]  #分离出特征和标签列

    ###这里需要对数据进行格式转换，转为序列的形式
    #tensor转numpy

    # 数据无量纲化处理
    x_data_set = StandardScaler().fit_transform(x_data_set)

    ####生成数据序列，每一个序列中报告N天的数据 csv文件中已按照序列的形式摆放，[0-30,31-61,...] 是一个序列
    x_data_seq = [x_data_set[i * config.time_step:i * config.time_step + config.time_step - 1, :] for i in
                  range(len(y_data_set) // config.time_step)]

    # 每个序列的标签 [序列数,]
    y_data_seq = [y_data_set[i * config.time_step + config.time_step - 1] for i in
                  range(len(y_data_set) // config.time_step)]

    x_data_seq = np.array(x_data_seq)
    y_data_seq = np.array(y_data_seq)
    # print(f"两个序列的shape:{x_data_seq.shape},{y_data_seq.shape}")

    x_data_seq = x_data_seq.reshape((x_data_seq.shape[0],-1))  #（序列数，历史天数，特征数）-》（序列数，自动调节）

    # 训练测试集分割
    train_x, test_x, train_y, test_y = train_test_split(x_data_seq, y_data_seq, random_state=config.random_seed,
                                                        shuffle=config.shuffle_train_data,
                                                        test_size=1 - config.train_data_rate)

    # 转为tensor类型
    tensor_train_x, tensor_train_y = torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    tensor_test_x, tensor_test_y =torch.FloatTensor(test_x),torch.FloatTensor(test_y)

    # 转换成torch的DataSet
    dataset_train = TensorDataset(tensor_train_x, tensor_train_y)  #转为TensorDataset
    dataset_test = TensorDataset(tensor_test_x, tensor_test_y)
    #创建随机数
    #drop_last 是否丢弃最后一个不完整的batch
    trainloader = DataLoader(dataset_train, config.batch_size, shuffle=config.shuffle_train_data)
    testloader = DataLoader(dataset_test, config.batch_size, shuffle=config.shuffle_train_data)

    return trainloader,testloader
def train_mpc(dataloader: DataLoader, model: torch.nn.Module, loss, config: Config):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)  # 优化器
    total_loss = None
    count = len(dataloader)
    model.train()
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        optimizer.zero_grad() #优化器清除梯度
        out = model(xs)  #前向传播
        loss_val = loss(out, ys)  #计算损失

        loss_val.backward()  #反向传播
        optimizer.step() #更新梯度

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()
    total_loss = total_loss.cpu().item()

    return total_loss / count
def validate_mpc(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module, config:Config):
    model.eval() #进入评估模式
    outs = []  #模型的输出结果
    true_lable_list = []  #真实的标签值
    total_loss = None
    count = len(dataloader)
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        out = model(xs)
        loss_val = loss(out, ys)  #损失值
        loss_val = loss_val.cpu()#切换到CPU上
        out = out.cpu() #切换到CPU上

        outs.append(out) #模型的输出结果
        true_lable_list.append(ys)  #真实的标签值

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.item()

    all_out = torch.cat(outs, dim=0)
    all_prob = all_out.cpu().detach().numpy()  #模型的输出值
    pred_ys = np.copy(all_prob)  #深拷贝
    pred_ys[all_prob >= config.good_value] = 1  # 按照阈值进行分类
    pred_ys[all_prob < config.good_value] = 0
    pred_ys = pred_ys.tolist()

    true_ys = torch.cat(true_lable_list, dim=0)
    true_ys = true_ys.cpu().numpy()  #测试数据集上的真实标签值
    # print("all_prob:",all_prob)
    # print("true_ys:",true_ys)
    return total_loss / count, roc_auc_score(true_ys, all_prob), precision_score(true_ys, pred_ys), recall_score(true_ys, pred_ys)

def display_result(history_result):
    epochs_train_loss, epochs_test_loss,epochs_test_auc, epochs_validate_precision, epochs_validate_recall = history_result
    # plt.subplot(1, 2, 1)  # 第一个图
    plt.plot(np.arange(len(epochs_train_loss)), epochs_train_loss, marker='s', linestyle='-', color='red',
             label='epochs_train_loss',linewidth=1.0,markersize=3)
    plt.plot(np.arange(len(epochs_test_loss)), epochs_test_loss, marker='s', linestyle='-',color='green',
             label='epochs_test_loss',linewidth=1,markersize=3)
    plt.xlabel("The epoch", fontsize=14)
    plt.ylabel("Loss value", fontsize=14)
    # plt.title("MLP Model Train Inform")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(fontsize=14)
    plt.show()
    # plt.subplot(1, 2, 2)  # 第二个图
    plt.plot(np.arange(len(epochs_test_auc)),epochs_test_auc,marker='s', linestyle='-',label='epochs_train_auc',
             linewidth=1,markersize=3)
    plt.xlabel("The epoch", fontsize=14)
    plt.ylabel("AUC value", fontsize=14)
    # plt.title("AUC of MLP Model")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(fontsize=14)
    plt.show()

    plt.plot(np.arange(len(epochs_validate_precision)),epochs_validate_precision,marker='s', linestyle='-',
             label='epochs_validate_precision',linewidth=1,markersize=3)
    plt.xlabel("The epoch", fontsize=14)
    plt.ylabel("precision_score", fontsize=14)
    # plt.title("AUC of MLP Model")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(fontsize=14)
    plt.show()

    plt.plot(np.arange(len(epochs_validate_recall)),epochs_validate_recall,marker='s', linestyle='-',
             label='epochs_validate_recall',linewidth=1,markersize=3)
    plt.xlabel("The epoch", fontsize=14)
    plt.ylabel("recall_score", fontsize=14)
    # plt.title("AUC of MLP Model")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(fontsize=14)
    plt.show()

def test_and_display_metrics(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module):
    model.eval()  # 进入评估模式
    outs = []  # 模型的输出结果
    true_lable_list = []  # 真实的标签值
    total_loss = None
    count = len(dataloader)
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        out = model(xs)
        loss_val = loss(out, ys)  # 损失值
        loss_val = loss_val.cpu()  # 切换到CPU上
        out = out.cpu()  # 切换到CPU上

        outs.append(out)  # 模型的输出结果
        true_lable_list.append(ys)  # 真实的标签值

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.item()

    all_out = torch.cat(outs, dim=0)
    all_prob = all_out.cpu().detach().numpy()#预测值
    all_prob[all_prob>1.0]=1.0
    all_prob[all_prob<0] = 0

    true_ys = torch.cat(true_lable_list, dim=0) #真实的标签值
    true_ys = true_ys.cpu().numpy()

    fpr_arr, tpr_arr, threshold_arr = metrics.roc_curve(true_ys, all_prob)  #用来绘制ROC曲线
    max_index=0
    max_youden_index = 0
    for i in range(len(fpr_arr)):  #寻找最佳的阈值
        youden_index = tpr_arr[i]-fpr_arr[i]
        if(youden_index > max_youden_index):
            max_youden_index = youden_index
            max_index = i
    good_value = threshold_arr[max_index]  #最佳的阈值
    print(f"最佳的阈值是:{good_value} 此时的约登指数为:{max_youden_index}\n")
    plt.plot(fpr_arr, tpr_arr, marker='s', linestyle='-', linewidth=1, markersize=3)  #绘制ROC曲线
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    # plt.title("AUC of MLP Model")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.legend(fontsize=14)
    plt.show()
    #


def main():
    config = Config()
    model = MLP(config)  # 创建模型 MLP  多层感知机

    # crypten.init()  #初始化环境
    # rank = comm.get().get_rank()
    # feature_name = names[rank] #获取特征名称

    train_dataloader,test_dataloader = make_dataloader(config)

    mpc_loss = torch.nn.MSELoss() #定义损失函数
    mpc_loss = mpc_loss.to(DEVICE)

    if config.do_train ==True :#训练模式

        model = model.to(DEVICE)
        epochs_train_loss = []
        epochs_validate_loss = []
        epochs_validate_auc = []
        epochs_validate_precision  = []
        epochs_validate_recall = []

        valid_loss_min = float("inf")  # 正无穷
        bad_epoch = 0
        for epoch in range(config.epoch):
            train_loss = train_mpc(train_dataloader, model, mpc_loss, config)
            if(train_loss<=0): #异常退出
                break
            epochs_train_loss.append(train_loss)

            validate_loss, auc, precision_score, recall_score = validate_mpc(test_dataloader, model, mpc_loss, config)
            epochs_validate_loss.append(validate_loss)
            epochs_validate_auc.append(auc)
            epochs_validate_precision.append(precision_score)
            epochs_validate_recall.append(recall_score)
            print(f"epoch: {epoch+1}/{config.epoch}, train loss: {train_loss}, validate loss: {validate_loss}, auc: {auc} "
                  f"precision_score:{precision_score} recall_score:{recall_score} \n\n")

            ##早挺机制
            if validate_loss < valid_loss_min:  # 存储最好的模型，如果当前的平均loss小于历史最小的loss
                valid_loss_min = validate_loss
                bad_epoch = 0
                torch.save(model.state_dict(), "./mpl_model_unvfl.pth")  # 模型保存
            else:
                bad_epoch += 1
                if bad_epoch >= config.patience:  # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                    print(" The training stops early in epoch {}".format(epoch))
                    break
        display_result((epochs_train_loss, epochs_validate_loss, epochs_validate_auc, epochs_validate_precision, epochs_validate_recall))
    else :  #非训练模式 指标可视化模模式

        model.load_state_dict(torch.load("./mpl_model_unvfl.pth"))  #加载模型,这个模型是训练好的最优模型  ，我们只用来测试各种指标 不训练
        # mpc_model = mpc_model.to(DEVICE)
        model = model.to(DEVICE)
        test_and_display_metrics(test_dataloader, model, mpc_loss) #测试模型并且生成各类指标的图像

if __name__ == '__main__':
    main()
