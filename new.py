import os
import pandas as pd
import numpy as np
from glob import glob

import xlwt
import xlrd

f = xlwt.Workbook()
# BOM
file_path1 = 'C://Users//76082//Desktop//123.xls'
# 属性表
file_path2 = 'C://Users//76082//Desktop//SU-MPU_V200_20210707.xls'
file_path3 = 'D://ku//*'
csv_list = glob(file_path3)  # 查看同文件夹下所有文件
file1 = pd.ExcelFile(file_path1)
file2 = pd.ExcelFile(file_path2)

for name in file1.sheet_names:
    file1 = pd.read_excel(file_path1, sheet_name=name, header=0)
    # sheet_names = file.sheet_names
    columns_id1 = file1.columns
    # print(columns_id1)
    datas1 = file1.values
    data1 = np.array(datas1)
    for m in range(columns_id1.shape[0]):
        if columns_id1[m] == 'Reference':
            ref1 = m
        elif columns_id1[m] == 'PART_NUMBER':
            pn1 = m
    # print(pn1, ref1)

for name in file2.sheet_names:
    file2 = pd.read_excel(file_path2, sheet_name=name, header=0)
    # sheet_names = file.sheet_names
    columns_id2 = file2.columns
    datas2 = file2.values
    data2 = np.array(datas2)
    for m in range(columns_id2.shape[0]):
        if columns_id2[m] == 'Part Reference':
            ref2 = m
        elif columns_id2[m] == 'PART_NUMBER':
            pn2 = m
    # print(pn2, ref2)

# news = {}
# for n in range(data1.shape[0]):
#     # print(data1[n][ref1])
#     news = data1[n][ref1].split(',')
#     pn_tem = data1[n][pn1]
#     for new in news:
#         for k in range(data2.shape[0]):
#             if data2[k][ref2] == new:
#                 data2[k][pn2] = pn_tem

sheet1 = f.add_sheet(sheetname="RK", cell_overwrite_ok=True)
labels = ['HEADER', '位号', 'PART_NUMBER', 'Part_Name', 'Value', 'PCB Footprint', 'Description', 'Library Source']
for n in range(data2.shape[0]):
    sheet1.write(n + 1, 2, data2[n][53])
    sheet1.write(n + 1, 3, "NC")
    sheet1.write(n + 1, 4, data2[n][3])
    sheet1.write(n + 1, 5, data2[n][55])
    sheet1.write(n + 1, 6, data2[n][16])
    sheet1.write(n + 1, 7, "NC")

for n in range(data2.shape[0]):
    for l in csv_list:
        file_name = os.path.basename(l)
        file_na = pd.ExcelFile(l)
        for name in file_na.sheet_names:
            file3 = pd.read_excel(l, sheet_name=name, header=0, keep_default_na=False)
            columns_id3 = file3.columns
            datas3 = file3.values
            data3 = np.array(datas3)
            for m in range(columns_id3.shape[0]):
                if columns_id3[m] == 'PART_NUMBER':
                    pn3 = m
            labels = np.array(labels)
            # labels = tem.append(labels)
            for label in range(labels.shape[0]):
                sheet1.write(0, label, labels[label])
            sheet1.write(n + 1, 0, data2[n][0])
            sheet1.write(n + 1, 1, datas2[n][ref2])
            print(data2[n][ref2])
            for d in range(data3.shape[0]):
                if data3[d][pn3] == data2[n][pn2]:
                    for e in range(1, 7):
                        sheet1.write(n + 1, e + 1, datas3[d][e])
            break
f.save('D:/result/result.xls')

# for label in range(labels.shape[0]):
#     sheet1.write(0, label, labels[label])
# for i in range(data2.shape[0]):
#     for k in range(data2.shape[1]):
#         if data2[i][k] is None:
#             sheet1.write(i + 1, k, np.NaN)
#         else:
#             sheet1.write(i + 1, k, data2[i][k])

# print('OK')
