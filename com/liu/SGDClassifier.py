from sklearn.linear_model import SGDClassifier
'''
可以通过partial_fit方法支持在线学习
'''
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
