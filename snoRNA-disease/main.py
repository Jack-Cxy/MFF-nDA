import time
import numpy as np
import pandas as pd
from utills import seed_everything
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from model import MLP
from metrics import get_metrics, get_metrics_original
seed_everything(42)


def cross_val(X, y, n_epochs, fold_num, learning_rate, in_dim_mlp, hidden_dim_mlp, dropout, device):

    # 定义训练过程
    def train_step(model, X_train, y_train):

        model.train()
        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train)

        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        print(logits)
        logits = logits.flatten()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.round(logits)
        correct = (predicted == labels).sum().item()
        acc = correct / labels.size(0)
        return loss, acc, logits

    # 定义验证过程
    def val_step(model, X_val, y_val):

        model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_val)
            labels = torch.from_numpy(y_val)

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            logits = logits.flatten()
            loss = loss_fn(logits, labels)

            predicted = torch.round(logits)
            correct = (predicted == labels).sum().item()
            acc = correct / labels.size(0)
        return loss, acc, logits

    # 定义五折交叉验证
    kfold = KFold(n_splits=fold_num, shuffle=True)
    metrics_avg_train = np.zeros((1, 7))
    metrics_avg_val = np.zeros((1, 7))

    for i, (train_index, val_index) in enumerate(kfold.split(X)):
        print("--" * 40)
        print('Fold:{}'.format(i + 1))

        # 划分训练集和验证集
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 初始化模型
        model = MLP(in_dim=in_dim_mlp, hidden_dim=hidden_dim_mlp, out_dim=1, dropout=dropout)
        model = model.to(device)
        loss_fn = nn.BCELoss()
        loss_fn = loss_fn.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 进行训练和验证

        for epoch in range(n_epochs):

            train_loss, train_acc, train_logits = train_step(model, X_train, y_train)
            val_loss, val_acc, val_logits = val_step(model, X_val, y_val)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f} '
                      f'Train acc: {train_acc:0.4f}'
                      f', Val acc: {val_acc:.4f}')

        print('--' * 40)
        metric_tmp_train = get_metrics_original(y_train, train_logits)
        metric_tmp_val = get_metrics_original(y_val, val_logits)

        print('Last Epoch Train: ', metric_tmp_train)
        print('Last Epoch Val: ', metric_tmp_val)

        metrics_avg_train += metric_tmp_train
        metrics_avg_val += metric_tmp_val

    print("--" * 40)
    print("Last Epoch Train 5cv avg: ", metrics_avg_train / 5)
    print("Last Epoch Val 5cv avg: ", metrics_avg_val / 5)


def main():
    stime = time.time()
    # gpu训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data = pd.read_csv(r'..\snoRNA-disease\data\result\feat_final_all.csv', index_col=0)
    print(data.shape)
    X = np.array(data.iloc[:, :-1], dtype='float32')
    y = np.array(data.iloc[:, -1], dtype='float32')
    # 超参数
    n_epochs = 300
    learning_rate = 0.001
    fold_num = 5
    hidden_dim_mlp = 1024
    dropout = 0.3

    in_dim_mlp = 357
    cross_val(X, y, n_epochs, fold_num, learning_rate, in_dim_mlp, hidden_dim_mlp, dropout, device)
    etime = time.time()
    print('-' * 10)
    print('Time:{:.4f}'.format(etime - stime))


if __name__ == '__main__':
    main()

