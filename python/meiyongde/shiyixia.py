import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pywt
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from python.meiyongde.maxmin import maxminnorm

if __name__ == '__main__':
    # 载入数据
    data1 = []
    label = []
    csv_list = glob('/python/meiyongde/anode/*')  # 查看同文件夹下的csv文件数
    for i in csv_list:
        file_name = os.path.basename(i.split('.')[0])
        data_read = pd.read_csv(i, header=None)
        data_read = np.array(data_read).T
        for j in range(data_read.shape[0]):
            data1.append(data_read[j])
            if int(file_name) < 15:
                label.append('0')
            elif int(file_name) > 119:
                label.append('2')
            else:
                label.append('1')
    data1 = np.array(data1)
    label = np.array(label)

    print(label)
    print(data1)

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(data1, label, random_state=5, test_size=0.2, shuffle=True)
    # 数据预处理
    # 归一化
    X_train = maxminnorm(X_train)
    X_test = maxminnorm(X_test)
    # DWT
    coeff = pywt.dwt(X_train, 'db3', mode='symmetric')
    X_train = coeff[0]
    coeff_test = pywt.dwt(X_test, 'db3', mode='symmetric')
    X_test = coeff_test[0]
    # # fft
    # X_train = fft(X_train).real
    # X_test = fft(X_test).real
    # mlp'identity', 'logistic', 'relu', 'softmax', 'tanh'
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver='adam', alpha=1e-4, random_state=3,
                        learning_rate='adaptive', learning_rate_init=0.001,
                        max_iter=500, tol=1e-4, verbose=0)

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 200
    N_BATCH = 4
    N_CLASSES = np.unique(y_train)
    scores_train = []
    scores_test = []
    scores = []
    scores_trains = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:

        # SHUFFLING
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
    # plt.plot(scores)
    # plt.show()

    # plt.plot(mlp.loss_curve_, color='blue', label='loss')
    # plt.title("loss over epochs", fontsize=14)
    # plt.legend(loc='upper right')
    # plt.show()

    """ Plot """
    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.show()
