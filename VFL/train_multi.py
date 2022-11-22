import sys

import crypten
import crypten.communicator as comm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_utils import crypten_collate
from models.lstm_model import Net,Config,load_data

names = ["a", "b", "c"]
feature_sizes = [50, 57, 1]

#读取本地numpy存储的文件生成Pytorch中的tensor类型
def load_local_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)  #使用numpy模型的文件读取功能，读取为一个numpy数组
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    tensor = torch.tensor(arr, dtype=torch.float32)  #转为pytorch 的tensor类型
    return tensor

#生成CrypTen中的cryptensor类型
def load_encrypt_tensor(filename: str) -> crypten.CrypTensor:
    local_tensor = load_local_tensor(filename)  #生成Pytorch 的Torch类型
    rank = comm.get().get_rank()  #获取当前客户端的编号
    count = local_tensor.shape[0]  #获取记录的行数

    encrypt_tensors = []  #存放每一个crypten类型的List
    for i, (name, feature_size) in enumerate(zip(names, feature_sizes)):  #根据客户端编号i也就是rank来加载不同的数据集
        if rank == i:
            assert local_tensor.shape[1] == feature_size, \
                f"{name} feature size should be {feature_size}, but get {local_tensor.shape[1]}"
            tensor = crypten.cryptensor(local_tensor, src=i)
        else:
            dummy_tensor = torch.zeros((count, feature_size), dtype=torch.float32)
            tensor = crypten.cryptensor(dummy_tensor, src=i)
        encrypt_tensors.append(tensor)

    res = crypten.cat(encrypt_tensors, dim=1)  #按照特征的维度拼接，即将不同的特征在组合为一个完整的样本
    return res



def make_local_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    tensor = load_local_tensor(filename)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader



#生成crypten类型的网络模型
def make_mpc_model(local_model: torch.nn.Module, config: Config):
    dummy_input = torch.empty((1, config.input_size))
    model = crypten.nn.from_pytorch(local_model, dummy_input)
    model.encrypt()
    return model


#生成dataloader
def make_mpc_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    mpc_tensor = load_encrypt_tensor(filename)  #将数据加载到cryptensor中
    feature, label = mpc_tensor[:, :-1], mpc_tensor[:, -1]  #分离出特征和标签列
    dataset = TensorDataset(feature, label)  #转为TensorDataset

    #创建随机数
    seed = (crypten.mpc.MPCTensor.rand(1) * (2 ** 32)).get_plain_text().int().item()
    generator = torch.Generator()
    generator.manual_seed(seed)

    #drop_last 是否丢弃最后一个不完整的batch
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=crypten_collate, generator=generator)
    return dataloader


def train_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module, config: Config):
    total_loss = None
    count = len(dataloader)

    model.train()
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        model.zero_grad()
        loss_val.backward()
        model.update_parameters(config.learning_rate)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()
    return total_loss / count


def validate_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module):
    model.eval()
    outs = []
    true_lable_list = []
    total_loss = None
    count = len(dataloader)
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)  #损失值

        outs.append(out) #模型的输出结果
        true_lable_list.append(ys)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()

    all_out = crypten.cat(outs, dim=0)
    all_prob = all_out.sigmoid()
    all_prob = all_prob.get_plain_text()
    pred_ys = torch.where(all_prob > 0.5, 1, 0).tolist()
    pred_probs = all_prob.tolist()

    true_ys = crypten.cat(true_ys, dim=0)
    true_ys = true_ys.get_plain_text().tolist()

    return total_loss / count, precision_score(true_ys, pred_ys), recall_score(true_ys, pred_ys), \
           roc_auc_score(true_ys, pred_probs)


def main():
    config = Config()
    model = Net(config)  # 创建模型 LSTM

    crypten.init()  #初始化环境

    rank = comm.get().get_rank()
    # rank =RANK

    name = names[rank]
    train_filename = f"dataset/{name}/train.npz"
    test_filename = f"dataset/{name}/test.npz"

    train_dataloader = make_mpc_dataloader(train_filename, config.batch_size, shuffle=True, drop_last=False)
    test_dataloader = make_mpc_dataloader(test_filename, config.batch_size, shuffle=False, drop_last=False)

    mpc_model = make_mpc_model(model,config)
    mpc_loss = crypten.nn.MSELoss() #定义损失函数

    for epoch in range(config.epoch):
        train_loss = train_mpc(train_dataloader, mpc_model, mpc_loss, config)
        print(f"epoch: {epoch+1}/{config.epoch}, train loss: {train_loss}")

        validate_loss, p, r, auc = validate_mpc(test_dataloader, mpc_model, mpc_loss)
        print(f"epoch: {epoch+1}/{config.epoch}, validate loss: {validate_loss}, precision: {p}, recall: {r}, auc: {auc}")


if __name__ == '__main__':
    main()
