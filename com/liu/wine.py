import  pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
'''
葡萄酒数据集
    将数据集划分为训练集和测试数据集
    从pandas库中可以直接获取开源的葡萄酒数据集
'''

'''
1.从机器学习样本库中在线读取葡萄酒数据集
'''
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

'''
2.将数据集划分为训练数据集和测试数据集 使用scikit-learn下model_selection模块的train_test_split
    在实际应用中，训练和测试集常用划分比例是6:4,7:3或8:2，对于非常庞大的数据集，9:1或者99:1也可以
    为了获取最佳的性能，完成分类模型的测试后，通常在整个数据集上需再次对模型进行训练
'''
X,y = df_wine.iloc[:, 1:], df_wine.iloc[:,0]
X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,test_size=0.3,random_state=0)
'''
3.将特征的值缩放到相同的区间
    决策树和随机森林是为数不多不需要进行特征缩放的算法。
    常用方法,具体语义由语境确定：
        1）归一化
             x(i)norm = (x(i)-x(min))/(x(max)-x(min))
            多数情况是将特征的值缩放到[0,1]，它是最小-最大缩放的一个特例
                在scikit-learn中使用MinMaxScaler()
            
        2）标准化
            x(i)std = (x(i) - 均值(x))/标准差(x)
            大部分机器学习中更加实用，保持了异常值所蕴含的有用信息，并且使得算法收到这些值的影响较小
'''
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)#对训练数据进行拟合及转换
X_test_norm = mms.transform(X_test)#对测试数据转换

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

'''
4.特征选择
    1.通过L1正则化稀疏模型参数:penalty='l1'
      lr.coef_:获取三个分类对应的权重系数向量
    2.序列后向选择算法(SBS,Sequential Backward Selection)
      1）将数据集压缩到低维的特征子空间
      2)步骤
            1.设k=d进行算法初始化，d是特征空间Xd的维度
            2.定义x_为满足标准x_argmaxJ(Xk-x)最大化的特征，其中x属于Xk
            3.将特征x_从特征集中删除:Xk-1 = Xk-x_,k=k-1
            4.如果k等于目标特征数量，算法终止，否则跳转第2步
      
'''
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(penalty='l1',C=0.1)
# lr.fit(X_train_std,y_train)
# print(lr.score(X_train_std,y_train))
#print(lr.intercept_)


'''
    使用实现的SBS应用于KNN分类器的效果
    可以提高测试机正确率，防止过拟合
'''
# from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt
# from com.liu.SBSal import SBS
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(X_train_std,y_train)
# print(knn.score(X_train_std,y_train))
# print(knn.score(X_test_std,y_test))
# sbs = SBS(knn,k_features=1)
# sbs.fit(X_train_std,y_train)
#
#
# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat,sbs.scores_,marker='o')
# plt.ylim([0.7,1.1])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
#plt.show()

#查看哪几个特征有比较好的表现
# print(sbs.subsets_)
# k5 = list(sbs.subsets_[8])
#print(df_wine.columns[1:][k5])

# knn.fit(X_train_std[:,k5],y_train)
# print(knn.score(X_train_std[:,k5],y_train))
# print(knn.score(X_test_std[:,k5],y_test))