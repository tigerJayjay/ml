import pandas as pd
from io import StringIO
csv_data = '''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,
'''
df = pd.read_csv(StringIO(csv_data))

'''
1.可以通过isnull().sum()来获取每一列中缺失的总数
2.DataFrame的values属性可以得到NumPy数组
3.通过dropna方法来删除数据集中包含缺失值的行,有以下参数值
    axis=1 删除至少包含一个NaN的列
    how='all'删除所有值都为NaN的行
    thresh=4 要求一行至少有4个非NaN才能不被删除
    subset=['C']只删除C列出现NaN的行
'''
print(df.isnull().sum())
df = df.dropna(thresh=2)
print(df)

'''
插值技术
    1.均值插补，通过特征均值来替换缺失值
        使用scikit-learn的Impute类
            axis=0 使用列均值 axis=1使用行均值
            stragegy还可以等于 median或者most_frequent
    2.Imputer
        fit方法用于对数据集中的参数进行识别并构建相应的数据补齐模型
        transform方法使用刚构建的模型对数据集中相应参数的缺失值进行补齐
'''
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN',strategy='mean',axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
#print(imputed_data)


'''
处理非数值型特征值
    1.标称特征:不具有顺序性，如颜色是
    2.有序特征：例如尺寸XL>L>M
'''
df = pd.DataFrame([
    ['green', 'M', 10.1,'class1'],
    ['red','L',13.5,'class2'],
    ['blue','XL',15.3,'class1']
])
df.columns = ['color','size','price','classlabel']
print(df)

'''
定义映射字典
    正向映射 df['columnName'].map(size_mapping)
    逆向映射 .................map(inv_size_mapping)
'''
size_mapping = {
    'XL' : 3,
    'L' :2,
    "M" : 1
}
df['size'] = df['size'].map(size_mapping)
print(df)

inv_size_mapping = {v:k for k,v in size_mapping.items()}
#print(inv_size_mapping)
#df['size'] = df['size'].map(inv_size_mapping)
#print(df)

'''
使用LabelEncoder类可以对类标进行整数编码
    fit_transform：相当于同时调用fit和transform
    inverse_transform将类标还原为初始字符串
'''
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
init_y  = class_le.inverse_transform(y)
print(init_y)

'''
对于标称特征（无序特征），如果编码成整数值，会变为有序特征，会导致结果不是最优的
    独热编码：将标称特征分称多个虚拟特征，比如color可以分为red,green,blue三个虚拟特征
             如果为蓝色那么blue=1 red =0 green=0
可以通过sklearn实现：OneHotEncoder
    通过设置categorical_features=[0]设置特征位置
    sparse=False返回一个常规的Numpy数组
可以通过pandas实现（更加方便) get_dummies()
'''
#先将color编码成整数
X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X))#稀疏矩阵
print(ohe.fit_transform(X).toarray())#转化为NumPy数组

pan = pd.get_dummies(df[['price','color','size']])
print(pan)
