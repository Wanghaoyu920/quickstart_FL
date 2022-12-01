import random
import sys
import crypten
import crypten.communicator as comm
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from data_utils import crypten_collate
from mlp_model import MLP,Config
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = ["气象", "地理加火灾标签"]
feature_sizes = [8,6]


def make_mpc_model(local_model: torch.nn.Module, config: Config):
    dummy_input = torch.empty((1, (config.time_step-1)*config.input_size))
    model = crypten.nn.from_pytorch(local_model, dummy_input)
    model.encrypt()
    return model

def load_csv_tensor(feature_name, config:Config)-> torch.Tensor:
    if feature_name=='气象':
        source_datafile = pd.read_csv("./data/田林县2016数据集_横向联邦_序列30_气象.csv",
                                      usecols=['PRS', 'TEM', 'RHU', 'PRE', 'WIN', 'WIN_Dir', 'SSD', 'GST'],
                                      dtype=np.float32, index_col=None, header=0)
    elif feature_name=='地理加火灾标签':
        source_datafile = pd.read_csv("./data/田林县2016数据集_横向联邦_序列30_地理加火灾标签.csv",
                                      usecols=['坡度', '坡向', 'EVI', '到河道距离', '到公路距离', 'Fire'],
                                      dtype=np.float32, index_col=None, header=0)
    else:
        print("特征名称不正确，无法打开正确的数据文件！")
        exit(-1)
    data_set = source_datafile.to_numpy()
    # print(f"源文件shape：{data_set.shape}")
    tensor = torch.tensor(data_set, dtype=torch.float32)  # 转为pytorch 的tensor类型
    return tensor


#补齐数据，并转为普通的tensor类型
def load_crypten_tensor(feature_name: str, config:Config)-> crypten.CrypTensor:
    local_tensor = load_csv_tensor(feature_name, config)  #生成Pytorch 的Torch类型,这个加载的数据一定是不全的
    rank = comm.get().get_rank()  # 获取当前客户端的编号
    count = local_tensor.shape[0]  # 获取记录的行数

    crypten_tensors = []  #存放每一个纵向联邦学习客户端的tensor
    for i, (name, feature_size) in enumerate(zip(names, feature_sizes)):  # 根据客户端编号i也就是rank来加载不同的数据集
        if rank == i:  #将数据加密为crypten形式
            tensor = crypten.cryptensor(local_tensor, src=i)  ## src标识数据的持有方
        else :
            tensor = torch.zeros((count, feature_size), dtype=torch.float32)# 数据补齐
            tensor = crypten.cryptensor(tensor, src=i)  #会来源与i的数据
        crypten_tensors.append(tensor)
    res = crypten.cat(crypten_tensors, dim=1)#按照特征的维度拼接，即将不同的特征在组合为一个完整的样本
    return res

#生成dataloader
def make_mpc_dataloader(feature_name: str, config:Config):
    # mpc_tensor = load_encrypt_tensor(feature_name, config)  #将数据加载到cryptensor中
    normal_tensor = load_crypten_tensor(feature_name, config)  #将数据加载到cryptensor中
    normal_tensor = normal_tensor.get_plain_text()  #获取为明文torch

    feature, label = normal_tensor[:, :-1], normal_tensor[:, -1]  #分离出特征和标签列

    ###这里需要对数据进行格式转换，转为序列的形式
    #tensor转numpy
    x_data_set = feature.numpy()
    y_data_set = label.numpy()

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

    # train_of_true_index = [random.randint(0,train_x.shape[0]-1) for i in range(int(train_x.shape[0]*0.2))]
    # test_of_true_index = [random.randint(0,test_y.shape[0]-1) for i in range(int(test_y.shape[0]*0.2))]
    # # ##控制数据量 总量大概是11W
    # train_x, test_x, train_y, test_y = train_x[train_of_true_index], test_x[test_of_true_index],\
    #                                    train_y[train_of_true_index], test_y[test_of_true_index]

    # print("train_y:", train_y)
    # print("test_y:", test_y)
    ######

    # 转为tensor类型
    tensor_train_x, tensor_train_y = crypten.cryptensor(torch.FloatTensor(train_x)), crypten.cryptensor(torch.IntTensor(train_y))
    tensor_test_x, tensor_test_y = crypten.cryptensor(torch.FloatTensor(test_x)), crypten.cryptensor(torch.IntTensor(test_y))

    # 转换成torch的DataSet
    dataset_train = TensorDataset(tensor_train_x, tensor_train_y)  #转为TensorDataset
    dataset_test = TensorDataset(tensor_test_x, tensor_test_y)


    #创建随机数
    seed = (crypten.mpc.MPCTensor.rand(1) * (2 ** 32)).get_plain_text().int().item()
    generator = torch.Generator()
    generator.manual_seed(seed)

    #drop_last 是否丢弃最后一个不完整的batch
    trainloader = DataLoader(dataset_train, config.batch_size, shuffle=config.shuffle_train_data,
                             collate_fn=crypten_collate, generator=generator)
    testloader = DataLoader(dataset_test, config.batch_size, shuffle=config.shuffle_train_data,
                             collate_fn=crypten_collate,generator=generator)

    return trainloader,testloader
def train_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module, config: Config):
    total_loss = None
    count = len(dataloader)

    model.train()
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        out = model(xs)
        loss_val = loss(out, ys)
        model.zero_grad()
        loss_val.backward()
        model.update_parameters(config.learning_rate)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()
    total_loss = total_loss.cpu()
    total_loss = total_loss.get_plain_text().item()

    return total_loss / count
def validate_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module, config:Config):
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

    total_loss = total_loss.get_plain_text().item()

    all_out = crypten.cat(outs, dim=0)
    all_prob = all_out.get_plain_text().cpu().numpy()  #模型的输出值
    pred_ys = np.copy(all_prob)  #深拷贝
    pred_ys[all_prob >= config.good_value] = 1  # 按照阈值进行分类
    pred_ys[all_prob < config.good_value] = 0
    pred_ys = pred_ys.tolist()

    true_ys = crypten.cat(true_lable_list, dim=0)
    true_ys = true_ys.get_plain_text().cpu().numpy()  #测试数据集上的真实标签值
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

def test_and_display_metrics(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module):
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

    total_loss = total_loss.get_plain_text().item()

    all_out = crypten.cat(outs, dim=0)
    all_out = all_out.get_plain_text()
    all_prob = all_out.cpu().numpy()#预测值
    all_prob[all_prob>1.0]=1.0
    all_prob[all_prob<0] = 0

    true_ys = crypten.cat(true_lable_list, dim=0) #真实的标签值
    true_ys = true_ys.get_plain_text().cpu().numpy()

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
    #目前CrypTen 只支持SGD优化器

    crypten.init()  #初始化环境
    rank = comm.get().get_rank()
    feature_name = names[rank] #获取特征名称

    train_dataloader,test_dataloader = make_mpc_dataloader(feature_name, config)

    mpc_loss = crypten.nn.MSELoss() #定义损失函数
    mpc_loss = mpc_loss.to(DEVICE)

    if config.do_train ==True :#训练模式
        mpc_model = make_mpc_model(model, config)  #加密模型
        # mpc_model = mpc_model.to(DEVICE)
        mpc_model = mpc_model.cuda()

        epochs_train_loss = []
        epochs_validate_loss = []
        epochs_validate_auc = []
        epochs_validate_precision  = []
        epochs_validate_recall = []


        valid_loss_min = float("inf")  # 正无穷
        bad_epoch = 0
        for epoch in range(config.epoch):
            train_loss = train_mpc(train_dataloader, mpc_model, mpc_loss, config)
            if(train_loss<=0): #异常退出
                break
            epochs_train_loss.append(train_loss)

            validate_loss, auc, precision_score, recall_score = validate_mpc(test_dataloader, mpc_model, mpc_loss, config)
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
                save_model =mpc_model.decrypt() #解密模型
                crypten.save(save_model.state_dict(), "./mpl_model_new.pth")  # 模型保存
                mpc_model.encrypt() #加密模型
            else:
                bad_epoch += 1
                if bad_epoch >= config.patience:  # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                    print(" The training stops early in epoch {}".format(epoch))
                    break
        display_result((epochs_train_loss, epochs_validate_loss, epochs_validate_auc, epochs_validate_precision, epochs_validate_recall))
    else :  #非训练模式 指标可视化模模式
        mpc_model = make_mpc_model(model, config)  # 生成crypten模型，并且加密模型
        mpc_model.decrypt() #解密
        mpc_model.load_state_dict(crypten.load("./mpl_model_new.pth"))  #加载模型,这个模型是训练好的最优模型  ，我们只用来测试各种指标 不训练
        mpc_model.encrypt()  # 加密模型
        # mpc_model = mpc_model.to(DEVICE)
        mpc_model = mpc_model.cuda()

        test_and_display_metrics(test_dataloader, mpc_model, mpc_loss) #测试模型并且生成各类指标的图像

if __name__ == '__main__':
    main()
