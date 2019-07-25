from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from com.liu.t import plot_decision_regions

'''
分类模型：逻辑斯谛回归
'''
iris = datasets.load_iris()
'''
    选择两个特征赋值给X
    将类标赋值给Y
'''
X = iris.data[:,[2,3]]
Y = iris.target

'''
    将数据分为训练数据集和测试数据集
    train_test_split函数，随机将数据矩阵X与类标向量y按照3：7的比例划分为测试数据集（45个样本）和训练数据集（105个样本）。
'''
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
'''
    使用StandardScaler的fit方法计算每个特征的样本均值和标准差
    使用transform方法可以使用样本均值和标准差对训练数据做标准化处理
'''
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#C是正则化系数的倒数
lr = LogisticRegression(C=1000.0,random_state=0)


lr.fit(X_train_std,y_train)
# X_combined_std = np.vstack((X_train_std,X_test_std))
# y_combined = np.hstack((y_train,y_test))
# plot_decision_regions(X_combined_std,y_combined,classifier=lr,
#                       test_idx=range(105,150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()
x_new = X_test_std[0,:];
x_new = np.array(x_new).reshape(1,-1)
print(x_new)
print(lr.predict_proba(x_new))

