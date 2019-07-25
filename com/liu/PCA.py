from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from com.liu.sklearntest import plot_decision_regions
from com.liu.wine import X_train_std,X_test_std,y_train,y_test
'''
主成分分析（principal component analysis，PCA)
    作用：降维
    算法流程：
        1.对原始d维数据集做标准化处理
        2.构造样本的协方差矩阵
        3.计算协方差矩阵的特征值和相应的特征向量
        4.选择与前k个最大特征值对应的特征向量，其中k为新特征空间的维度(k<=d)
        5.通过前k个特征向量构建映射矩阵W
        6.通过映射矩阵W将d维的输入数据集X转换到新的k维特征子空间
        使用PCA可以在忽略类标的情况下，将数据映射到一个低维的子空间上
            并沿正交的特征坐标方向使方差最大化
'''
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
lr1 = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
print(X_train_pca)
X_test_pca = pca.transform(X_test_std)
lr1.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier=lr1)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

#查看测试集的分类情况
plot_decision_regions(X_test_pca,y_test,classifier=lr1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

