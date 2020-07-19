import torch
import sys
import pywt
import numpy as np

from numpy.fft import fft

# 数据装载
import xlwt
from sklearn.model_selection import train_test_split

from python.read_file import read_current_class

X_train, X_test, y_train, y_test = read_current_class()
_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)


# fft
# X_train = fft(X_train).real
# X_val = fft(X_val).real
# X_test = fft(X_test).real

# data_train = torch.from_numpy(X_train[1])


def data_transform_dwt(arr):
    arr = arr / 1667
    coeff = pywt.dwt(arr, 'db3', mode='symmetric')
    datas = coeff[0]
    datas = torch.from_numpy(datas)
    return datas


def data_transform_fft(arr):
    arr = arr / 1667
    datas = fft(arr).real
    datas = torch.from_numpy(datas)
    return datas


arrs = data_transform_dwt(X_test)
arrs2 = data_transform_fft(X_test)


class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(100, 70, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(70),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(70, 140, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(140),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(140 * 2, 800),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(800, 5))

    # 前向传播
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 140 * 2)
        x = self.dense(x)
        return x


class CNN_Model2(torch.nn.Module):
    def __init__(self):
        super(CNN_Model2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(100, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(128 * 4, 800),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(800, 5))

    # 前向传播
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, x2.shape[1] * x2.shape[2])
        x = self.dense(x)
        return x


cl = 0
hj = 0
hl = 0
xy = 0
zc = 0
cl1 = 0
hj1 = 0
hl1 = 0
xy1 = 0
zc1 = 0
# 模型加载
# dwt
model = torch.load('C:/Users/76082/PycharmProjects/bigdata/python/data/cnn_model2.pth')
outputs = model(arrs.float())
_, pre = torch.max(outputs, 1)
print(len(pre.numpy()))

for im in range(len(pre.numpy())):
    if y_test[im] == 0:
        cl += 1
        if pre.numpy()[im] == y_test[im]:
            cl1 += 1
    elif y_test[im] == 1:
        hj += 1
        if pre.numpy()[im] == y_test[im]:
            hj1 += 1
    elif y_test[im] == 2:
        hl += 1
        if pre.numpy()[im] == 2:
            hl1 += 1
    elif y_test[im] == 3:
        xy += 1
        if pre.numpy()[im] == 3:
            xy1 += 1
    elif y_test[im] == 4:
        zc += 1
        if pre.numpy()[im] == 4:
            zc1 += 1
print('出铝状况准确率为：' + str(cl1/cl))
print('换极状况准确率为：' + str(hj1/hj))
print('阳极滑落状况准确率为：' + str(hl1/hl))
print('阳极效应状况准确率为：' + str(xy1/xy))
print('正常状况准确率为：' + str(zc1/zc))

# fft
model1 = torch.load('C:/Users/76082/PycharmProjects/bigdata/python/data/cnn_model1.pth')
outputs = model1(arrs2.float())
_, pre = torch.max(outputs, 1)
cl = 0
hj = 0
hl = 0
xy = 0
zc = 0
cl1 = 0
hj1 = 0
hl1 = 0
xy1 = 0
zc1 = 0
# 模型加载
# dwt

for im in range(len(pre.numpy())):
    if y_test[im] == 0:
        cl += 1
        if pre.numpy()[im] == y_test[im]:
            cl1 += 1
    elif y_test[im] == 1:
        hj += 1
        if pre.numpy()[im] == y_test[im]:
            hj1 += 1
    elif y_test[im] == 2:
        hl += 1
        if pre.numpy()[im] == 2:
            hl1 += 1
    elif y_test[im] == 3:
        xy += 1
        if pre.numpy()[im] == 3:
            xy1 += 1
    elif y_test[im] == 4:
        zc += 1
        if pre.numpy()[im] == 4:
            zc1 += 1
print('出铝状况准确率为：' + str(cl1/cl))
print('换极状况准确率为：' + str(hj1/hj))
print('阳极滑落状况准确率为：' + str(hl1/hl))
print('阳极效应状况准确率为：' + str(xy1/xy))
print('正常状况准确率为：' + str(zc1/zc))
# print(pre)

# f = xlwt.Workbook()
# sheet1 = f.add_sheet(sheetname="c220220160614", cell_overwrite_ok=True)
# for i in range(X_train[37].shape[0]):
#     for j in range(X_train[37].shape[1]):
#         sheet1.write(i, j, float(X_train[1149][i][j]))
# f.save('D:/jiashuju/c220220160614.xls')
