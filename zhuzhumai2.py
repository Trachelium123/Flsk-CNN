import os
import pandas as pd
from glob import glob

import xlwt
import numpy as np

labels = ['订单号', '商品名称', '订单创建时间', '运费', '订单实付金额', '课程简称', '收货人/提货人', '收货人手机号/提货人手机号',
          '详细收货地址/提货地址', '买家备注', '买家姓名', '买家手机号', '商家订单备注', '订单商品状态', '商品类型', '商品类目', '商品规格',
          '商品编码', '商品单价', '商品数量', '商品留言', '商品发货物流公司', '商品发货物流单号', '商品退款状态', '商品已退款金额']
f = xlwt.Workbook()
sheet1 = f.add_sheet(sheetname="zhuzhumai", cell_overwrite_ok=True)
for zhuzhumai in range(len(labels)):
    sheet1.write(0, zhuzhumai, labels[zhuzhumai])
csv_list = glob('D:/file/*')  # 查看同文件夹下所有文件
datastyle = xlwt.XFStyle()
datastyle.num_format_str = 'yyyy-mm-dd hh:mm'
m = 0
for l in csv_list:
    file_name = os.path.basename(l)
    file_na = pd.ExcelFile(l)
    for name in file_na.sheet_names:
        file = pd.read_excel(l, sheet_name=name, header=0, keep_default_na=False)
        data = np.array(file)
        for i in range(data.shape[0]):
            m += 1
            for k in range(data.shape[1]):
                if k == 2:
                    sheet1.write(m, k, data[i][k], datastyle)
                else:
                    sheet1.write(m, k, data[i][k])
f.save('D:/result.xls')
