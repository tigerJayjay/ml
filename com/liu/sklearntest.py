from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from com.liu.t import plot_decision_regions
import matplotlib.pyplot as plt
'''
    感知器
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

'''
    训练感知器模型，将数据同时输入到感知器
'''
ppn = Perceptron(max_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

'''
    使用predict进行预测
'''
y_pred = ppn.predict(X_test_std)
print((y_test != y_pred).sum())

'''
    使用accuracy_socre计算感知器在测试数据集上的分类准确率
'''
print(accuracy_score(y_test,y_pred))

'''
    图像展示
'''
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()