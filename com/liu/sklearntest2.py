from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from com.liu.t import plot_decision_regions

'''
支持向量机(support vector machine,SVM)
偏差：欠拟合，训练不足，数据的扰动不会对结果产生很大的影响，这个时候偏差主导了算法的泛化能力
方差：随着训练的进行，学习器的拟合能力逐渐增强，偏差逐渐减小，但此时通过不同数据学习的学习器就会有较大的片擦，此时方差主导模型的泛化能力
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

#kernel=linear线性
#kernel='rbf'非线性
#gamma的值越小，会导致决策边界更加宽松
svm = SVC(kernel='rbf',C=1.0,random_state=0,gamma=100)
svm.fit(X_train_std,y_train)
print(X_train_std,X_test_std)
print('----')
X_combined_std = np.vstack((X_train_std,X_test_std))
print(X_combined_std)
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined_std,y_combined,classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

'''
逻辑斯谛回归会尽量最大化训练数据集的条件似然，这使得它比支持向量机更易于处理离群点，而支持向量机则更关注接近决策边界的点，
逻辑斯蒂谛回归模型简单更容易实现，更新方便，当应用于流数据分析时，这是非常具备吸引力的
'''
