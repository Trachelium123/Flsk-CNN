import csv
import glob
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np


def do_fft(y):
    # 采样点选择
    x = range(0, len(y))
    # 傅里叶变换
    fft_y = fft(y)
    # 取绝对值
    y_abs = abs(fft_y)
    # 归一化处理
    y_to = y_abs / len(x)
    # 由于对称性，只取一半区间
    y_sub = y_to[range(int(len(x) / 2))]
    xf = np.arange(len(x))  # 频率
    xf1 = xf
    xf2 = xf[range(int(len(x) / 2))]  # 取一半区间

    plt.subplot(221)
    plt.plot(x, y)
    plt.title('Original wave')

    plt.subplot(222)
    plt.plot(xf, y_abs, 'r')
    plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表

    plt.subplot(223)
    plt.plot(xf1, y_to, 'g')
    plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')

    plt.subplot(224)
    plt.plot(xf2, y_sub, 'b')
    plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')

    plt.show()


csv_list = glob.glob('C:/Users/76082/PycharmProjects/bigdata/demo/*')  # 查看同文件夹下的csv文件数
for i in csv_list:
    with open(i, 'r') as file:
        file_name = os.path.basename(i.split('.')[1])
        if file_name == 'csv':
            reader = csv.reader(file)
            for row in reader:
                j = 0
                while j < len(row):
                    row[j] = float(row[j])
                    j += 1
                do_fft(row)
