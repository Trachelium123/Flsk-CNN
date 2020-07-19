import os
import pandas as pd
import numpy as np
import pickle

import pywt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt


# cl 出铝 => 0
# hj 换极 => 1
# hl 阳极滑落 => 2
# xy 阳极效应 => 3
# zc 正常 => 4
def read_current_class():
    path = u"D:/异常数据整理/异常数据/单工况"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    data = []
    label = []
    for i, file in enumerate(files):  # 遍历文件夹
        file = path + "/" + file
        if os.path.isfile(file):  # 判断是文件,才打开
            file = open(file)  # 中文文件名先执行这条命令
            ct = pd.read_csv(file, header=None, dtype=np.float32).values
            ct = ct/1667
            # ct = maxminnorm(ct)
            [num, dim] = ct.shape
            num = int(np.floor(num / 100))
            for j in range(num):
                label.append(i)
                data.append(ct[j * 100:j * 100 + 100, :])
    # data = np.mat([item for sublist in data for item in sublist])
    data = np.array(data)
    # label = np.mat([item for sublist in label for item in sublist])
    label = np.array(label).flatten()
    print(label.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test
    # return data,label


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_current_class()
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    # 特征提取
    # DWT
    coeff = pywt.dwt(X_train, 'db3', mode='symmetric')
    X_train = coeff[0]
    print(y_train.shape)

    coeff_test = pywt.dwt(X_test, 'db3', mode='symmetric')
    X_test = coeff_test[0]

    mlp = MLPClassifier(hidden_layer_sizes=[400, 200, 100, 50], activation="relu", solver='adam',
                        alpha=1e-4, random_state=5, learning_rate='adaptive',
                        learning_rate_init=0.001, max_iter=500, tol=1e-4, verbose=0)

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 200
    N_BATCH = 64
    N_CLASSES = np.unique(y_train)
    scores_train = []
    scores_test = []
    scores = []
    scores_trains = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:

        # SHUFFLING,随机排序,（0-1174）
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))
        # scores_trains.append(cross_val_score(mlp, X_train, y_train, scoring='accuracy', cv=5).mean())
        # SCORE TEST
        scores_test.append(mlp.score(X_test, y_test))
        print('epoch: ', epoch, '=>', mlp.best_loss_)

        # nn_predict = mlp.predict(data1)
        # scores.append(accuracy_score(nn_predict, label))
        epoch += 1
        # 保存模型
    file1 = 'C:/Users/76082/PycharmProjects/bigdata/model/model.pickle'
    with open(file1, 'wb') as f:
        pickle.dump(mlp, f)

    plt.plot(mlp.loss_curve_, color='blue', label='loss')
    plt.title("loss over epochs", fontsize=14)
    plt.legend(loc='upper right')
    plt.show()

    """ Plot """
    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.show()
