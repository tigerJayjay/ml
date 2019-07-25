from sklearn.base import clone
from itertools import  combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
序列后向选择算法（Sequence Backward Selection)
    用作特征选择
'''
class SBS():
    def __init__(self,estimator,k_features,
                 scoring=accuracy_score,
                 test_size=0.25,
                 random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self,X,y):
        X_train,X_test,y_train,y_test=\
                train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        dim = X_train.shape[1]#当前特征数
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]#子特征数
        score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_)
        self.scores_ = [score]#获取初始分值
        while dim>self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_,r=dim-1):#对当前特征数进行特征数-1的排列组合，获取到子序列
                score = self._calc_score(X_train,y_train,
                                         X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best] #获取到当前子序列中分值最高的特征值子序列
            self.subsets_.append(self.indices_)#获取当前数量分值最高的子序列
            dim -= 1
            self.scores_.append(scores[best])#获取当前最好的分值

        self.k_score_ = self.scores_[-1]#获取分值最高的分数
        return self

    def transform(self,X):
        return X[:,self.indices_]

    def _calc_score(self,X_train,y_train,X_test,y_test,indices):
        self.estimator.fit(X_train[:,indices],y_train)#通过给定的机器学习算法，训练模型
        y_pred = self.estimator.predict(X_test[:,indices])#通过测试集得到预测结果
        score = self.scoring(y_test,y_pred)#获取到预测分值
        return score