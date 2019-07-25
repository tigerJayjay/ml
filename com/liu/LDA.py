from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from com.liu.wine import X_train_std,X_test_std,y_train,y_test
from sklearn.linear_model import LogisticRegression
from com.liu.sklearntest import plot_decision_regions
import matplotlib.pyplot as plt

'''
线性判别分析压缩无监督数据
与PCA相比,LDA是一种更优越的用于分类的特征提取技术
是一种监督降维技术，在线性特征空间中尝试使得类别最大可分时，需要使用
    训练数据集中的类别信息
'''
lda = LinearDiscriminantAnalysis(n_components=2)
print(X_train_std)
X_train_lda = lda.fit_transform(X_train_std,y_train)
lr = LogisticRegression()
lr = lr.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

#显示模型在测试数据集上的效果
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda,y_test,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()