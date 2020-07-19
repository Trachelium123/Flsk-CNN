import os
import pandas as pd
import numpy as np
import xlwt

f = xlwt.Workbook()
file_path = 'D:/047f6681e7622425.xlsx'
file_name = os.path.basename(file_path)
file = pd.ExcelFile(file_path)
# 获取工作表名称
for name in file.sheet_names:
    file = pd.read_excel(file_path, sheet_name=name, header=0)
    # sheet_names = file.sheet_names
    datas = file.loc[:, ['订单创建时间', '订单号', '商品名称', '商品规格', '商家订单备注', '商品单价']].values
    labels = ['下单时间', '订单号', '课程名称', '课程规格', '兑换码', '售价', '课程简称']
    data = np.array(datas)
    for m in range(data.shape[0]):
        if data[m][4].find('失败') > -1:
            data[m][4] = data[m][4]
        elif data[m][4].find('成功') > -1 or data[m][4].find('已发卡密') > -1:
            print(data[m][4])
            data[m][4] = data[m][4].split(':')[1]

    sheet1 = f.add_sheet(sheetname=name, cell_overwrite_ok=True)
    for j in range(len(labels)):
        sheet1.write(0, j, labels[j])
    for i in range(data.shape[0]):
        for k in range(data.shape[1]):
            sheet1.write(i + 1, k, data[i][k])
        sheet1.write(i + 1, data.shape[1], name)
f.save('D:/result.xls')
