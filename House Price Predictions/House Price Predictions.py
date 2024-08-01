import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv(r"C:\Users\user\OneDrive\デスクトップ\py\home-data-for-ml-course\train.csv")
df_test = pd.read_csv(r"C:\Users\user\OneDrive\デスクトップ\py\home-data-for-ml-course\test.csv")
df_train.head()
df_test.head()

r,c = df_train.shape #shape用于返回 DataFrame 的维度。
#r 将存储 DataFrame 的行数。c 将存储 DataFrame 的列数。
print('The training data has {} rows and {} columns'.format(r,c))
r,c = df_test.shape
print('The validation data has {} rows and {} columns'.format(r,c))
#.format(r, c)：用于将变量的值插入到字符串的占位符中。
#The training data has 1460 rows and 81 columns
#The validation data has 1459 rows and 80 columns

df_train.info()#用于获取DataFrame的简要摘要信息。
#索引范围和列数。每列的名称、非空值的数量和数据类型。DataFrame的内存使用情况
#尤其看非空值数量

plt.figure(figsize=(24,8))
# 空值最多的列
#isnull().sum()布尔值计算每列中缺失值的数量。
#sort_values对Series进行排序。ascending=False表示按降序排列，即从缺失值最多的列到缺失值最少的列进行排序。
cols_with_null=df_train.isnull().sum().sort_values(ascending=False)
# 绘制缺失值最多的列的柱状图可视化
sns.barplot(x=cols_with_null.index,y=cols_with_null)
plt.xticks(rotation=90)
plt.show()

cols_with_null.head(10)
#cols_to_drop=(cols_with_null.head(6).index).tolist()
#df_train.drop(cols_to_drop,axis=1,inplace=True)
#df_test.drop(cols_to_drop,axis=1,inplace=True)
#df_train.shape
df_train['SalePrice'].isnull().sum()

df_train.head()
df_train.describe()

#A.简要列出一些特点
important_features=['YearBuilt','LotArea','OverallQual','OverallCond','GrLivArea','1stFlrSF','2ndFlrSF','BedroomAbvGr','OpenPorchSF','PoolArea','SalePrice']
df_train[important_features].describe()

import seaborn as sns
import pandas as pd
    

plt.figure(figsize=(15,12))
numeric_df = df_train.select_dtypes(include=['number'])# 选择数值类型的列
sns.heatmap(numeric_df.corr())#绘制热力图，用于查看变量之间的相关性。df_train.corr()计算了各列之间的相关系数矩阵。
#相关系数矩阵是一个对称矩阵，对角线上的值为1，表示每列与自身的相关性为100%。热力图通过颜色深浅来表示相关系数的大小，颜色越深表示相关性越强。
plt.show()

#删除不重要的项目
un_imp=['MSSubClass','OverallCond','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','3SsnPorch','YrSold','MoSold','MiscVal','PoolArea']
#df_train.drop(un_imp,axis=1,inplace=True)

#外部检测
fig, ax=plt.subplots(1,3,figsize=(28,7))
sns.scatterplot(x=df_train.GrLivArea,y=df_train.SalePrice,size=df_train.BedroomAbvGr,hue=df_train.OverallQual, ax=ax[0])
ax[0].set_title("Ground Living Area")#设置了子图的标题
sns.scatterplot(x=df_train.LotArea,y=df_train.SalePrice,size=df_train.BedroomAbvGr,hue=df_train.OverallQual, ax=ax[1])
ax[1].set_title("LOT AREA")#设置了子图的标题
sns.boxplot(x=df_train.SalePrice);
plt.show()
#地面居住面积超过 4000 平方米的房屋为离群值。
#地块面积超过 6000 的房屋为离群值。
#售价超过 45000 的房屋会影响我们的模型，因为其中大部分都是离群值。

sns.catplot(data=df_train, y='SalePrice', x='OverallQual', kind="boxen"); #多列条形图。
plt.show()
df_train['SalePrice'].quantile(0.995)#计算并返回 SalePrice 列（房屋销售价格）的 99.5% 分位点的值。
#对于识别高价房或潜在的离群值非常有用，帮助分析和处理异常数据。
#527331.9149999974

rows_2_drop=df_train[df_train['SalePrice']>df_train['SalePrice'].quantile(0.995)].index
df_train.drop(rows_2_drop,inplace=True)
df_train.shape
#(1452, 81)
rows_2_drop=df_train[df_train['GrLivArea']>4000].index
df_train.drop(rows_2_drop,inplace=True)
df_train.shape
#(1450, 81)
df_train[df_train['LotArea']>45000]
rows_2_drop=df_train[df_train['LotArea']>45000].index
df_train.drop(rows_2_drop,inplace=True)
df_train.shape
#(1439, 81)

#为建模准备数据
X_train = df_train.drop(['Id','SalePrice'],axis=1)
y_train = df_train.SalePrice
X_test = df_test.drop(['Id'],axis=1)

# 选择分类数据
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

# 选择数值数据
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# 数值数据的预处理 - 缺失值用常数填补
numerical_transformer = SimpleImputer(strategy='constant')#填补缺失值

# 分类数据的预处理 - 缺失值用众数填补，独热编码
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

# 对数字和分类数据进行捆绑预处理
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

#建模 梯度提升回归模型GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
model_GBR =  GradientBoostingRegressor(n_estimators=1100, loss='squared_error', subsample = 0.35, learning_rate = 0.05,random_state=1)
GBR_Pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model_GBR)])#管道化处理
GBR_Pipeline.fit(X_train, y_train)
preds_GBR = GBR_Pipeline.predict(X_test)

#提交
submission= pd.DataFrame({'Id': df_test.Id,'SalePrice': preds_GBR})
submission.head()
submission.to_csv('submission.csv',index=False)
