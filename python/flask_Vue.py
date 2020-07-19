import numpy as np
import pywt
import torch
from flask import Flask, request
from numpy.fft import fft

from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


def data_transform_dwt(arr):
    coeff = pywt.dwt(arr, 'db3', mode='symmetric')
    datas = coeff[0]
    datas = torch.from_numpy(datas)
    return datas


def data_transform_fft(arr):
    datas = fft(arr).real
    datas = torch.from_numpy(datas)
    return datas


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


@app.route('/dwt', methods=['GET', 'POST', 'PUT'])
# 按钮指向的路由
def predict():
    # name = request.form.get("name", )
    arrs = request.json["name"]
    arrs = np.array(arrs)
    arrs = arrs / 1667
    # 模型加载
    model = torch.load('C:/Users/76082/PycharmProjects/bigdata/python/data/cnn_model2.pth')
    outputs = model(data_transform_dwt(arrs).float().unsqueeze(0))
    _, pre = torch.max(outputs, 1)
    return str(pre.numpy()[0])


@app.route('/fft', methods=['GET', 'POST', 'PUT'])
# 按钮指向的路由
def predict2():
    # name = request.form.get("name", )
    arrs = request.json["name"]
    arrs = np.array(arrs)
    arrs = arrs / 1667
    # 模型加载
    model = torch.load('C:/Users/76082/PycharmProjects/bigdata/python/data/cnn_model1.pth')
    outputs = model(data_transform_fft(arrs).float().unsqueeze(0))
    _, pre = torch.max(outputs, 1)
    return str(pre.numpy()[0])


if __name__ == '__main__':
    app.run(port=5000,)
