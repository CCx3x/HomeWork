import xlrd
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import norm

mydata = xlrd.open_workbook("./作业数据.xls")
mysheet1 = mydata.sheet_by_name("Sheet1")

# 获取行数、列数
nRows = mysheet1.nrows
nCols = mysheet1.ncols

# 用于存取男生女生身高数据
man_height = []
woman_height = []

# 获取第4列的内容:身高
for i in range(nRows):
    if i + 1 < nRows:
        if mysheet1.cell(i + 1, 1).value == 1:
            man_height.append(mysheet1.cell(i + 1, 3).value)
        elif mysheet1.cell(i + 1, 1).value == 0:
            woman_height.append(mysheet1.cell(i + 1, 3).value)

# 获取男、女生的数量
manlen = len(man_height)
womanlen = len(woman_height)

# 画男女生身高频谱图
plt.hist(man_height, manlen, align='left', color='red', label='boy')
plt.hist(woman_height, womanlen, align='right', label='girl')
plt.legend(loc=0)
plt.xlabel('height')
plt.xlim(min(man_height + woman_height) - 1, max(man_height + woman_height) + 1)
plt.ylabel('number')
plt.title('Boy height spectrum')
# xsticks与yticks：指定坐标轴的刻度
plt.xticks(np.arange(min(man_height + woman_height), max(man_height + woman_height) + 1, 1.0))
plt.yticks(np.linspace(0, 50, 26))
plt.show()
