from glob import glob
import os
import pickle
import numpy as np
import pandas as pd

from numpy.fft import fft

from python.meiyongde.maxmin import maxminnorm

if __name__ == '__main__':
    data1 = []
    label = []
    scores = []
    csv_list = glob('/python/meiyongde/anode/*')
    for i in csv_list:
        file_name = os.path.basename(i.split('.')[0])
        data_read = pd.read_csv(i, header=None)
        data_read = np.array(data_read).T
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

    data1 = maxminnorm(data1)
    # coeff = pywt.dwt(data1, 'db3', mode='symmetric')
    # data1 = coeff[0]

    data1 = fft(data1).real


    file1 = 'C:\\Users\\76082\\PycharmProjects\\bigdata\\model\\model.pickle'
    k=0
    with open(file1, 'rb')as f:
        model = pickle.load(f)
        predict = model.predict(data1)
        for i in range(0, len(predict)):
            if predict[i] != label[i]:
                k+=1
    print(k/len(predict))
    #     scores.append(accuracy_score(predict, label))
    # print(scores)




