import numpy as np
import os
import pickle
import pywt
import pywt.data
import pandas as pd

# 获取样本矩阵的特征向量
from glob import glob

from sklearn.metrics import accuracy_score


def WaveletAlternation(SingleSample_Data):
    Featureweidu, SingleDir_Samples = SingleSample_Data.shape  # 获取矩阵的列数和行数，即样本维数 24 * 100
    SingleDir_SamplesFeature = np.zeros((Featureweidu, 8))  # 定义样本特征向量 #Array 形式
    #      SingleDir_SamplesFeature = [] # list形式
    for i in range(Featureweidu):
        SingleSampleDataWavelet = SingleSample_Data[i, :]  # 对第i行做小波包分解
        # 进行小波变换，提取样本特征
        wp = pywt.WaveletPacket(SingleSampleDataWavelet, wavelet='db3', mode='symmetric', maxlevel=3)  # 小波包三层分解
        # print([node.path for node in wp.get_level(3, 'natural')])   #第3层有8个
        # 获取第level层的节点系数
        aaa = wp['aaa'].data  # 第1个节点
        aad = wp['aad'].data  # 第2个节点
        ada = wp['ada'].data  # 第3个节点
        add = wp['add'].data  # 第4个节点
        daa = wp['daa'].data  # 第5个节点
        dad = wp['dad'].data  # 第6个节点
        dda = wp['dda'].data  # 第7个节点
        ddd = wp['ddd'].data  # 第8个节点
        # 求取节点的范数
        ret1 = np.linalg.norm(aaa, ord=None)  # 第一个节点系数求得的范数/ 矩阵元素平方和开方
        ret2 = np.linalg.norm(aad, ord=None)
        ret3 = np.linalg.norm(ada, ord=None)
        ret4 = np.linalg.norm(add, ord=None)
        ret5 = np.linalg.norm(daa, ord=None)
        ret6 = np.linalg.norm(dad, ord=None)
        ret7 = np.linalg.norm(dda, ord=None)
        ret8 = np.linalg.norm(ddd, ord=None)
        # 8个节点组合成特征向量
        SingleSampleFeature = [ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]
        # print(SingleSampleFeature)
        SingleDir_SamplesFeature[i][:] = SingleSampleFeature  # Array 形式
    #            SingleDir_SamplesFeature.append(SingleSampleFeature)   #list 形式
    #            print('SingleDir_SamplesFeature:', SingleDir_SamplesFeature)
    return SingleDir_SamplesFeature


if __name__ == '__main__':
    data1 = []
    label = []
    scores = []
    csv_list = glob('/python/meiyongde/anode/*')
    for i in csv_list:
        file_name = os.path.basename(i.split('.')[0])
        data_read = pd.read_csv(i, header=None)
        data_read = np.array(data_read).T
        data_read = WaveletAlternation(data_read)
        print(data_read.shape)
        for j in range(data_read.shape[0]):
            data1.append(data_read[j])
            if int(file_name) < 15:
                label.append('0')
            elif int(file_name) > 100:
                label.append('2')
            else:
                label.append('1')

    data1 = np.array(data1)
    label = np.array(label)

    # a = WaveletAlternation(np.array(data1[0]).reshape(1, 100))
#     print(np.array(data1[0]).shape)
# file1 = 'D:/jar/model/model.pickle'
# with open(file1, 'rb')as f:
#     model = pickle.load(f)
#     predict = model.predict(a)
#     print(predict)
