import math
import matplotlib.pyplot as plt
import pywt
import torch
import torch.utils.data as dataf
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from numpy.fft import fft
from sklearn.model_selection import train_test_split

from python.read_file import read_current_class

acc = []
acc2 = []
acc1 = []
loss_data = []
loss_data1 = []
num_epochs = 50
batch_size = 64
learning_rate = 0.001


# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


# 对数据进行载入及有相应变换,将Compose看成一种容器，他能对多种数据变换进行组合
# 传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, ], std=[0.5, ])])

# 数据装载
X_train, X_test, y_train, y_test = read_current_class()
_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

# fft
X_train = fft(X_train).real
X_val = fft(X_val).real
X_test = fft(X_test).real

data_train = torch.from_numpy(X_train)
data_val = torch.from_numpy(X_val)
data_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)
y_test = torch.from_numpy(y_test)
data_train = dataf.TensorDataset(data_train, y_train)
data_val = dataf.TensorDataset(data_val, y_val)
data_test = dataf.TensorDataset(data_test, y_test)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_val = torch.utils.data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)


class CNN_Model2(torch.nn.Module):
    def __init__(self):
        super(CNN_Model2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(100, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(128 * 4, 800),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(800, 5))

    # 前向传播
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, x2.shape[1] * x2.shape[2])
        x = self.dense(x)
        return x

# 对模型进行训练和参数优化
cnn_model = CNN_Model2()
# 将所有的模型参数移动到GPU上
if torch.cuda.is_available():
    cnn_model = cnn_model.cuda()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0.0
    running_loss1 = 0.0
    print("Epoch  {}/{}".format(epoch, num_epochs))
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = get_variable(X_train), get_variable(y_train)
        outputs = cnn_model(X_train.float())
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = loss_func(outputs, y_train.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    val_correct = 0.0
    for data in data_loader_val:
        X_val, y_val = data
        X_val, y_val = get_variable(X_val), get_variable(y_val)
        outputs = cnn_model(X_val.float())
        _, pred = torch.max(outputs, 1)  # 返回每一行中最大值的那个元素，且返回其索引
        optimizer.zero_grad()
        loss = loss_func(outputs, y_val.long())
        loss.backward()
        optimizer.step()
        running_loss1 += loss.item()
        val_correct += torch.sum(pred == y_val.data)
    testing_correct = 0.0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = get_variable(X_test), get_variable(y_test)
        outputs = cnn_model(X_test.float())
        _, pred = torch.max(outputs, 1)  # 返回每一行中最大值的那个元素，且返回其索引
        testing_correct += torch.sum(pred == y_test.data)
        # print(testing_correct)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,val Accuracy is:{:.4f}%".format(
        running_loss / len(data_train), 100 * running_correct / len(data_train),
        100 * val_correct / len(data_val)))
    acc.append(running_correct / len(data_train))
    acc1.append(val_correct / len(data_val))
    acc2.append(testing_correct / len(data_test))
    loss_data.append(running_loss / len(data_train))
    loss_data1.append(running_loss1 / len(data_val))

plt.plot(acc, color='green', alpha=0.8, label='Train')
plt.plot(acc1, color='blue', alpha=0.8, label='Val')
plt.title("Accuracy over epochs", fontsize=14)
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

plt.plot(acc2, color='magenta', alpha=0.8, label='Test')
plt.title("Accuracy over epochs", fontsize=14)
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_data, color='magenta', alpha=0.8, label='Train')
plt.plot(loss_data1, color='blue', alpha=0.8, label='Val')
plt.title("Loss over epochs", fontsize=14)
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()
# 保存模型
torch.save(cnn_model, 'cnn_model1.pth')
# 加载模型
cnn_model = torch.load('cnn_model1.pth')
