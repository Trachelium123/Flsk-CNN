import glob
import os
import pandas as pd
import numpy as np
import xlwt
f = xlwt.Workbook()
csv_list = glob.glob('D:/jiashuju/*')  # 查看同文件夹下的csv文件数
for j in csv_list:
    file_name = os.path.basename(j)
    print(file_name)
    file = pd.read_excel(j, header=0)
    data = np.array(file)
    sheet1 = f.add_sheet(sheetname=str(file_name), cell_overwrite_ok=True)
    for m in range(data.shape[1]):
        sheet1.write(1, m, str('f' + str(m)))
    for i in range(data.shape[0]):
        for k in range(data.shape[1]):
            sheet1.write(i+1, k, float(data[i][k]))
f.save('D:/bigdata.xls')
