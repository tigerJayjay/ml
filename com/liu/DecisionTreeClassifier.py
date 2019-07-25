from sklearn.tree import  DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from com.liu.t import plot_decision_regions

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

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)
tree.fit(X_train,y_train)

X_combined = np.vstack((X_train,X_test))

y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()