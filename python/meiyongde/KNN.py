import csv
import glob
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from python.meiyongde.maxmin import maxminnorm

if __name__ == '__main__':

    # 载入数据
    data = []
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
                data.append(row)
                if int(file_name) < 15:
                    label.append('0')
                elif int(file_name) > 100:
                    label.append('2')
                else:
                    label.append('1')

    data = np.array(data)
    label = np.array(label)

    # 数据预处理
    # 归一化
    data = maxminnorm(data)
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=4, shuffle=True)

    # # 建立测试参数集
    # k_range = range(1, 31)
    # k_scores = []
    # # 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, X_test, y_test, cv=10, scoring='accuracy')
    #     k_scores.append(scores.mean())
    # print(scores)
    # # 可视化数据
    # plt.plot(k_range, k_scores)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.show()

    # 模型定义
    # 1.KNN模型
    knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
    # 模型拟合
    knn.fit(X_train, y_train)
    # 模型预测
    knn_predict = knn.predict(X_test)
    # 基础打分
    knn_score = knn.score(X_test, y_test)
    print(knn_score)

    # 模型评估
    # 交叉验证
    knn_cross = cross_val_score(knn, X_test, y_test, scoring='accuracy', cv=10, n_jobs=-1)
    print(knn_cross)
    print(knn_cross.mean())

    # 保存模型
    with open('model/model_knn.pickle', 'wb') as f:
        pickle.dump(knn, f)
    # 加载模型
    with open('model/model_knn.pickle', 'rb')as f:
        model = pickle.load(f)

    datas = [[0, 2, 4, 6, 8, 2, 5, 6, 1, 2, 3, 3, 45, 2, 0.1, 12, 23, 0.3, 0.36, 0.56, 0.32, 0.21, 0.596, 0.785]]
    model_predict = model.predict(datas)
    print(model_predict)
