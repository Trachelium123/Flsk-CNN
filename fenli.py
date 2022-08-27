import pandas as pd
import numpy as np
import xlwt

file_path3 = 'C://Users//76082//Desktop//ZHKJ_CIS.xls'
file3 = pd.ExcelFile(file_path3)
for name in file3.sheet_names:
    file3 = pd.read_excel(file_path3, sheet_name=name, header=0,keep_default_na=False)
    columns_id3 = file3.columns
    datas3 = file3.values
    data3 = np.array(datas3)
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheetname="RK", cell_overwrite_ok=True)
    columns_id3 = file3.columns
    labels = np.array(columns_id3)
    for label in range(labels.shape[0]):
        sheet1.write(0, label, labels[label])
    for i in range(data3.shape[0]):
        for k in range(data3.shape[1]):
            if data3[i][k] is np.NaN:
                sheet1.write(i + 1, k, '')
            else:
                sheet1.write(i + 1, k, data3[i][k])
    na = 'D:/ku/' + name + '.xls'
    f.save(na)
