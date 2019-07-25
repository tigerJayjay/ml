import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from sklearn.linear_model import  LogisticRegression
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import cross_val_score
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
X = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
#使用Pipeline集成操作
pipe_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])
pipe_lr.fit(X_train,y_train)
print(pipe_lr.score(X_test,y_test))

'''
交叉验证评估模型性能
    为了在偏差和方差之间找到折中方案
        1.holdout交叉验证
            1）为了验证模型在测试数据上的性能，一般会重复使用测试数据进行调参，但是这样
            测试数据也就被当成训练数据，这样会导致模型过拟合
            2）将数据划分为三个部分：训练数据集，验证数据集，测试数据集
                训练数据集用于不同模型的拟合
                在验证数据集上的性能表现作为模型选择的标准
                
        2.k折交叉验证
            1）将训练数据集划分为k个，k-1个用于模型的训练,剩余一个用于测试，重复k次得到k个模型和对
                模型性能的评价
            2）对数据划分方法的敏感性较低，可以用于模型的调优，一旦找到满意的超参值，可以在全部的
                训练数据上重新训练模型，并使用独立的测试数据集对模型做出最终的评价
            3）k的标准值为10，如果训练数据集较小，加大k值，反之减小k值
            4）分层k折交叉验证
                1）类别比例在每个分块中都与训练数据集的整体比例一致
            5)sklearn中分层k折交叉验证的实现
                cross_val_score:
                    estimator 流水线
                    cv 划分数量
                    n_jobs 将不同分块的性能评估分不到多个cpu上进行处理 -1所有cpu
            
'''
scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=2)
#print(scores)




'''
通过学习及验证曲线来调试算法
    学习曲线
        样本大小与准确率之间的关系
    验证曲线
        准确率与参数之间的关系
'''

#使用scikit-learn中的学习曲线函数评估模型
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
pipe_lr = Pipeline([
    ('scl',StandardScaler()),
    ('clf',LogisticRegression(
        penalty='l2',random_state=0
    ))
])
'''
np.linspace 均分0.1到1之间的数 分为十个
np.mean()求取均值，以m*n矩阵为例
    axis不设置，返回m*n个数的均值
    axis=0:压缩行，求取各列均值， 返回1*n矩阵
    axis=1:压缩列，求取各行均值，返回m*1矩阵
np.std()计算标准差
plt.legend 曲线标识显示位置
'''
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=X_train,y=y_train, train_sizes=np.linspace(0.1,1.0,10),
                                                    cv=10,n_jobs=2)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5, label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5, label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std, alpha='0.5',color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()

#通过验证曲线来判定过拟合与欠拟合
from sklearn.model_selection import validation_curve
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
train_scores,test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='clf__C',
    param_range=param_range,
    cv=10
)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5, label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5, label='validation accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std, alpha='0.5',color='green')
plt.grid()
plt.xscale('log')
#plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.show()