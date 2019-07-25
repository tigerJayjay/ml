from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
'''
核PCA可以将非线性数据集映射到一个低维的特征空间中，使得数据线性可分
'''
X,y = make_moons(n_samples=100, random_state=123)
print(X)
print(y)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
print(y==0)
plt.scatter(X_skernpca[y==0,0], X_skernpca[y==0,1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1,0], X_skernpca[y==1,1],
            color='blue',marker='o',alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()