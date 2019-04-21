import xlrd
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import norm

'''
/**************************task1**************************/
Take the height as an example, draw a histogram of the 
height of the boys and girls and compare
/**************************task1**************************/
'''
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

'''
/**************************task2**************************/
Using the maximum likelihood estimation method to find the 
parameters of height and weight distribution for boys and girls
/**************************task2**************************/
'''
# 用于存取男生女生体重数据
man_weight = []
woman_weight = []

# 将男女生体重数据从第5列中进行分离，并保存在上述空数组中
for i in range(nRows):
    if i + 1 < nRows:
        if mysheet1.cell(i + 1, 1).value == 1:
            man_weight.append(mysheet1.cell(i + 1, 4).value)
        elif mysheet1.cell(i + 1, 1).value == 0:
            woman_weight.append(mysheet1.cell(i + 1, 4).value)
# fit(data):Return MLEs for shape, location, and scale parameters from data
# norm.fit(x)就是将x看成是某个norm分布的抽样，求出其最好的拟合参数（mean, std）
man_height_mean, man_height_std = norm.fit(man_height)  # 男生升高分布参数
man_weight_mean, man_weight_std = norm.fit(man_weight)  # 男生体重分布参数
woman_height_mean, woman_height_std = norm.fit(woman_height)  # 女生升高分布参数
woman_weight_mean, woman_weight_std = norm.fit(woman_weight)  # 女生体重分布参数

man_height_variance = man_height_std ** 2
man_weight_variance = man_weight_std ** 2
woman_height_variance = woman_height_std ** 2
woman_weight_variance = woman_weight_std ** 2
print(man_height_mean, man_height_variance, man_weight_mean, man_weight_variance)
print(woman_height_mean, woman_height_variance, woman_weight_mean, woman_weight_variance)

'''
/**************************task3**************************/
采用贝叶斯估计方法，求男女生身高以及体重分布的参数（注明自己选定的参数情况）
/**************************task3**************************/
'''
man_height_mean_cntr, man_height_var_cntr, man_height_std_cntr = st.bayes_mvs(man_height)
man_weight_mean_cntr, man_weight_var_cntr, man_weight_std_cntr = st.bayes_mvs(man_weight)
woman_height_mean_cntr, woman_height_var_cntr, woman_height_std_cntr = st.bayes_mvs(woman_height)
woman_weight_mean_cntr, woman_weight_var_cntr, woman_weight_std_cntr = st.bayes_mvs(woman_weight)


# print(man_height_mean_cntr.statistic,man_weight_mean_cntr.statistic)
# print(woman_height_mean_cntr.statistic,woman_weight_mean_cntr.statistic)

def get_mean_bayes(arr, mean0, variance0, variance):
    datasum = sum(arr)
    datalen = len(arr)
    mean_bayes = (variance0 * datasum + variance * mean0) / (datalen * variance0 + variance)
    return mean_bayes


'''
特殊情况1:
variance0=0时，mean_bayes=mean0
先验知识可靠，样本不起作用
#以男生身高为例，选定参数：mean0=20,variance=10,variance0=0
'''
man_height_mean_bayes = get_mean_bayes(man_height, 20, 0, 10)
print(man_height_mean_bayes)

'''
特殊情况2:
variance0>>variance时，mean_bayes=sample_mean
先验知识十分不确定，完全依靠样本信息,结果与最大似然估计结果近似
#以女生身高为例，选定参数：mean0=50,variance=1,variance0=100
'''
woman_height_mean_bayes = get_mean_bayes(woman_height, 50, 100, 1)
print(woman_height_mean_bayes)

'''
/**************************task4**************************/
4.	采用最小错误率贝叶斯决策，画出类别判定的决策面。并判断某样本
的身高体重分别为(160,45)时应该属于男生还是女生？为(178,70)时呢？
/**************************task4**************************/
'''


# ①本题输入类数为2：即男生、女生；特征数为2：即身高、体重
# ②求协方差矩阵
def get_covariance_matrix_coefficient(arr1, arr2):  # arr1与arr2长度相等
    datalength1 = len(arr1)
    datalength2 = len(arr2)
    sum_temp = []
    for i in range(datalength1):
        sum_temp.append((arr1[i] - sum(arr1) / datalength1) * (arr2[i] - sum(arr2) / datalength2))
        c12 = sum(sum_temp)
    covariance_matrix_c12 = c12 / (datalength1 - 1)
    return covariance_matrix_c12


man_c11 = man_height_variance
man_c22 = man_weight_variance
man_c12 = man_c21 = get_covariance_matrix_coefficient(man_height, man_weight)
man_covariance_matrix = np.matrix([[man_c11, man_c12], [man_c21, man_c22]])
woman_c11 = woman_height_variance
woman_c22 = woman_weight_variance
woman_c12 = woman_c21 = get_covariance_matrix_coefficient(woman_height, woman_weight)
woman_covariance_matrix = np.matrix([[woman_c11, woman_c12], [woman_c21, woman_c22]])
print(woman_covariance_matrix)

# 求男生、女生先验概率
man_priori_probability = manlen / (manlen + womanlen)
woman_priori_probability = 1 - man_priori_probability
# print(woman_priori_probability)

man_feature_mean_vector = np.matrix([[man_height_mean], [man_weight_mean]])
woman_feature_mean_vector = np.matrix([[woman_height_mean], [woman_weight_mean]])


# 定义等高线高度函数
def f(sample_height, sample_weight):
    mytemp1 = np.zeros(shape=(100, 100))
    for i in range(100):
        for j in range(100):
            sample_vector = np.matrix([[sample_height[i, j]], [sample_weight[i, j]]])
            sample_vector_T = np.transpose(sample_vector)
            # 定义决策函数
            mytemp1[i, j] = 0.5 * np.transpose(sample_vector - man_feature_mean_vector) * (
                np.linalg.inv(man_covariance_matrix)) * \
                            (sample_vector - man_feature_mean_vector) - 0.5 * np.transpose(
                sample_vector - woman_feature_mean_vector) * \
                            (np.linalg.inv(woman_covariance_matrix)) * (sample_vector - woman_feature_mean_vector) + \
                            0.5 * math.log(
                (np.linalg.det(man_covariance_matrix)) / (np.linalg.det(woman_covariance_matrix))) - \
                            math.log(man_priori_probability / woman_priori_probability)
    return mytemp1


sample_height = np.linspace(150, 180, 100)
sample_weight = np.linspace(40, 80, 100)
# 将原始数据变成网格数据
Sample_height, Sample_weight = np.meshgrid(sample_height, sample_weight)
# 填充颜色
plt.contourf(Sample_height, Sample_weight, f(Sample_height, Sample_weight), 0, alpha=0)
# 绘制等高线,圈内为女生，圈外为男生
C = plt.contour(Sample_height, Sample_weight, f(Sample_height, Sample_weight), 0, colors='black', linewidths=0.6)
# 显示各等高线的数据标签
plt.clabel(C, inline=True, fontsize=10)

# 显示男女生样本散点图

p1 = plt.scatter(man_height, man_weight, c='g', marker='*', linewidths=0.4)
p2 = plt.scatter(woman_height, woman_weight, c='r', marker='*', linewidths=0.4)


# 定义显示坐标函数
def Display_coordinates(m, n):
    plt.scatter(m, n, marker='s', linewidths=0.4)
    plt.annotate((m, n), xy=(m, n))
    return


# 并判断某样本的身高体重分别为(160,45)时应该属于男生还是女生？为(178,70)时呢
Display_coordinates(160, 45)
Display_coordinates(178, 70)
label = ['boy', 'girl']
plt.legend([p1, p2], label, loc=0)
plt.xlabel('height/cm')
plt.ylabel('weight/kg')
plt.show()

