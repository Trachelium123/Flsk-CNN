import csv
import os
import glob
import os, random, shutil
import pandas as pd

# 拆分文件
csv_list = glob.glob('/python/meiyongde/anode/*')  # 查看同文件夹下的csv文件数
for i in csv_list:  # 循环读取同文件夹下的csv文件
    file_name = os.path.basename(i.split('.')[0])
    with open(i, 'r') as file:
        reader = csv.reader(file)
        j = 0
        for row in reader:
            with open('C:/Users/76082/PycharmProjects/bigdata/datas/' + file_name + '_' + str(j) + '.csv', 'a') as f:
                k = 0
                while k < len(row):
                    if k == 23:
                        f.write(row[k])
                    else:
                        f.write(row[k] + ',')
                    k += 1
            j += 1


# # 深度学习过程中，需要制作训练集和验证集、测试集。
# def moveFile(fileDir):
#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#     filenumber = len(pathDir)
#     rate = 1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#     picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
#     sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
#     print(sample)
#     for name in sample:
#         shutil.move(fileDir + name, tarDir + name)
#     return


# if __name__ == '__main__':
#     fileDir = "C:/Users/76082/PycharmProjects/bigdata/datas/"  # 源图片文件夹路径
#     tarDir = 'C:/Users/76082/PycharmProjects/bigdata/val/'  # 移动到新的文件夹路径
#     moveFile(fileDir)

# #打标签
# csv_list = glob.glob('C:/Users/76082/PycharmProjects/bigdata/test/*')  # 查看同文件夹下的csv文件数
# for i in csv_list:
#     file_name_number = os.path.basename(i.split('_')[0])
#     file_name_number = int(file_name_number)
#     file_name = os.path.basename(i)
#     with open('C:/Users/76082/PycharmProjects/bigdata/test/test.txt', 'a') as f:
#         if file_name_number < 11:
#             f.write('C:/Users/76082/PycharmProjects/bigdata/test/' + file_name + ',0\n')
#         elif file_name_number > 100:
#             f.write('C:/Users/76082/PycharmProjects/bigdata/test/' + file_name + ',2\n')
#         else:
#             f.write('C:/Users/76082/PycharmProjects/bigdata/test/' + file_name + ',1\n')
