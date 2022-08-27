import os
import pandas as pd
import numpy as np
from glob import glob

import xlwt
import xlrd

f = xlwt.Workbook()
file_path1 = 'D://RK3308.xlsx'
file_path2 = 'D://bom.xlsx'
file_path3 = 'D://ku//*'
csv_list = glob(file_path3)  # 查看同文件夹下所有文件
file1 = pd.ExcelFile(file_path1)
file2 = pd.ExcelFile(file_path2)

for name in file1.sheet_names:
    file1 = pd.read_excel(file_path1, sheet_name=name, header=0)
    # sheet_names = file.sheet_names
    columns_id1 = file1.columns
    datas1 = file1.values
    data1 = np.array(datas1)
    for m in range(columns_id1.shape[0]):
        if columns_id1[m] == 'Reference':
            ref1 = m
        elif columns_id1[m] == 'P/N':
            pn1 = m
    # print(pn1, ref1)
