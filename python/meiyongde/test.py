import csv
import glob
import pickle

import pywt

import numpy as np
import os

import sklearn.model_selection
import sklearn.neural_network as sk_nn

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # 载入数据
    data1 = []
    label = []
    csv_list = glob.glob('/python/meiyongde/anode/*')  # 查看同文件夹下的csv文件数
    for i in csv_list:
        file_name = os.path.basename(i.split('.')[0])
        with open(i, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                j = 0
                while j < len(row):
                    row[j] = float(row[j])
                    j += 1
                data1.append(row)
                if int(file_name) < 15:
                    label.append('0')
                elif int(file_name) > 100:
                    label.append('2')
                else:
                    label.append('1')

    data1 = np.array(data1)
    label = np.array(label)
    # 数据预处理

    # dwt变换
    coeff = pywt.dwt(data1, 'db3', mode='symmetric')
    data = np.array(coeff[0])

    # fft
    # data = fft(data1).real

    # KSVD
    # scores1 = []
    # scores2 = []
    # for i in range(15, 30):
    # ksvd = KSVD(24)
    # dictionary, sparsecode = ksvd.fit(data1)
    # data = dictionary.dot(sparsecode)

    # 数据拆分
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, random_state=3,
                                                                                test_size=0.25, shuffle=True)

    # nn
    nn = sk_nn.MLPClassifier(activation='relu', solver='adam', alpha=0.0001, batch_size=8,
                             learning_rate='adaptive', learning_rate_init=0.001, max_iter=400, tol=1e-8, power_t=0.5)
    # 模型拟合
    nn.fit(X_train, y_train)
    # 模型预测
    nn_predict = nn.predict(X_test)
    # 模型评估
    # 基础打分
    nn_score = nn.score(X_test, y_test)
    print(nn_score)
    print(nn.n_outputs_)  # 输出类别数
    print(nn.classes_)  # 所有类别
    print("loss:", nn.loss_)  # 损失函数的损失值
    # print(nn.intercepts_)  # 偏移量
    # print(nn.coefs_)  # 权重
    print("迭代次数:", nn.n_iter_)  # 迭代轮数
    print(nn.n_layers_)  # 网络层数，只有一层隐藏层时 =3
    print(nn.out_activation_)  # 输出层激活函数的名称
    print("预测正确率为：", accuracy_score(nn_predict, y_test))
    print(nn.out_activation_)

    # 交叉验证
    # nn_cross1 = cross_val_score(nn, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1)
    # nn_cross2 = cross_val_score(nn, X_test, y_test, scoring='accuracy', cv=10, n_jobs=-1)
    # print(nn_cross1)
    # print(nn_cross2)
    # #     scores1.append(nn_cross1.mean())
    # #     scores2.append(nn_cross2.mean())
    # #     print(nn_cross1.mean())
    # #     print(nn_cross2.mean())
    # # plt.plot(scores1, linestyle='-', color='r', label='train')
    # # plt.plot(scores2, linestyle='-', color='b', label='test')
    # # plt.show()
    # 校验曲线

    # train_score, test_score = validation_curve(nn, X_train, y_train, param_name="max_iter",
    #                                            param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)

    # sklearn2pmml(nn, 'C:\\Users\\76082\\PycharmProjects\\bigdata\\test_model.pmml')
    # joblib.dump(nn, "C:/Users/76082/PycharmProjects/bigdata/model/model.pkl.z", compress=9)

    # print(test_score)

    # 保存模型
    with open('../../model/model.pickle', 'wb') as f:
        pickle.dump(nn, f)
    # # 加载模型
    # with open('model_ksvd.pickle', 'rb')as f:
    #     model = pickle.load(f)
    # # 使用训练好的模型
    # datas = [
    #     [0.3625, 0.5615, 0.32, 0.21, 0.596, 0.785, 0.569, 0.2154, 0.6985, 0.485, 0.7526, 0.2365, 0.8965, 0.46698, 0.215,
    #      0.596, 0.785, 0.569, 0.2154, 0.6985, 0.485, 0.3698, 0.1458, 0.2852]
    # ]
    # model_predict = model.predict(datas)
    # print(model_predict)
