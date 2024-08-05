#提升洞察力：利用 XGBoost 进行保险交叉销售
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

#阅读和理解我们的数据
original_train = pd.read_csv('/Users/lulei/Desktop/机器学习/playground-series-s4e7/archive/train.csv')
train = pd.read_csv("/Users/lulei/Desktop/机器学习/playground-series-s4e7/train.csv")
test = pd.read_csv('/Users/lulei/Desktop/机器学习/playground-series-s4e7/test.csv')
original_train.head()
train.head()
original_train.shape
train.shape
train.dtypes#查看列名和数据类型。
#根据 "id "列计算数据帧 df 中的重复行数。
len(train[train.duplicated(['id'])])
len(original_train[train.duplicated(['id'])])
#id 列中的所有值都是唯一的，删除 id 列：
train.drop('id', axis=1, inplace=True)
original_train.drop('id', axis=1, inplace=True)

#检查是否有缺失值。
#使用带有 "海绿 "色的浅色调色板创建自定义色彩图  
cmap = sns.light_palette("seagreen", as_cmap=True)
#创建热图，直观显示主 DataFrame 中的缺失值数量
plt.figure(figsize=(max(22, 11), 4))
sns.heatmap(
    (train.isna().sum()).to_frame(name='').T, 
    cmap=cmap, 
    annot=True, 
    fmt='0.0f'
).set_title('Count of Missing Values in the main dataset', fontsize=18)
plt.show()
#创建热图，直观显示原始 DataFrame 中缺失值的数量
plt.figure(figsize=(max(22, 11), 4))
sns.heatmap(
    (original_train.isna().sum()).to_frame(name='').T, 
    cmap=cmap, 
    annot=True, 
    fmt='0.0f'
).set_title('Count of Missing Values in original dataset', fontsize=18)
plt.show()

#检查分类列中的唯一值及其比例计数：
train.Gender.value_counts(normalize=True)
train.Vehicle_Age.value_counts(normalize=True)
train.Vehicle_Damage.value_counts(normalize=True)

#描述性统计
train.describe().iloc[1:]
original_train.describe().iloc[1:]
#让我们来看看那些没有驾驶执照的人：
train[train.Driving_License == 0].sample(20)
train[train.Driving_License == 0].Gender.value_counts()
train[train.Driving_License == 0].Age.value_counts().head(10)
train.Region_Code.nunique()#Region_Code 列中，有 54 个不同的唯一值。

#初步数据探索总结
#训练数据集包含超过 1 150 万行和 12 列，所有值均为非空。数据集包含整数、浮点数和字符串。以下是主要观察结果：
#性别和车辆损坏是二元分类。相比之下，车辆年龄有三个类别。值得注意的是，只有 4.1% 的汽车车龄超过 2 年。
#年龄：20 至 85 岁，平均 38 岁。
#曾有保险：46% 以前投过保。
#年保费：2.6 至 54 万，平均 3 万。
#驾照：99.8% 的人有驾照。
#拥有汽车但没有驾驶执照的大多数人似乎都是 60 岁以上的男性。这可能表明，他们为自己的子女或孙子购买了汽车，但以自己的名义注册，或由他人代为驾驶。
#地区代码（Regional_Code）是一个分类变量，但我们将其保留为数值，以避免产生 54 个额外列。

#内存优化策略
#我们实现了一个函数，用于将数据类型转换为更节省内存的替代类型
#定义函数 converting_datatypes(df):
def converting_datatypes(df):
    df = df.copy()
    try:
        # Converting data types
        df['Gender'] = df['Gender'].astype('category')
        df['Vehicle_Age'] = df['Vehicle_Age'].astype('category')
        df['Vehicle_Damage'] = df['Vehicle_Damage'].astype('category')
        df['Age'] = df['Age'].astype('int8')
        df['Driving_License'] = df['Driving_License'].astype('int8')
        df['Region_Code'] = df['Region_Code'].astype('int8')
        df['Previously_Insured'] = df['Previously_Insured'].astype('int8')
        df['Annual_Premium'] = df['Annual_Premium'].astype('int32')
        df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int16')
        df['Vintage'] = df['Vintage'].astype('int16')
        df['Response'] = df['Response'].astype('int8')
        print(df.info(memory_usage='deep'))
    except KeyError as e:
        print(f"Error: {e} not found in DataFrame")
    except Exception as e:
        print(f"An error occurred: {e}")
    return df
#转换数据类型
train = converting_datatypes(train)
original_train = converting_datatypes(original_train)
test = converting_datatypes(test)
#查看数据类型
train.dtypes
original_train.dtypes
test.dtypes
#查看内存优化结果
train_memory_usage = train.memory_usage(deep=True).sum()
original_train_memory_usage = original_train.memory_usage(deep=True).sum()
test_memory_usage = test.memory_usage(deep=True).sum()
# 计算所有 DataFrame 的总内存使用量
total_memory_usage = train_memory_usage + original_train_memory_usage + test_memory_usage
print(f"所有 DataFrame 的总内存使用量: {total_memory_usage / 1024 ** 2:.2f} MB")
#采用优化策略后，训练集的内存使用量下降
#这表明内存使用量大幅减少，除了我们正在处理的数据量外，还使数据集更易于分析和建模。

#数据可视化
#探索相关性
# 计算数据帧的相关矩阵
z = train.corr(numeric_only=True)
#创建带有文字注释和自定义色阶的交互式热图
fig = px.imshow(
    z, 
    text_auto=True, 
    aspect="auto", 
    color_continuous_scale="blugrn"
)
#更新布局，加入自定义样式的标题
fig.update_layout(
    title="Correlation Heatmap",
    title_font_size=18,
    title_x=0.5,  # Center the title
    title_y=0.95  # Slightly adjust the title position
)
#显示交互式热图
fig.show()
df_corr = train.corr(numeric_only=True)['Response'][:-1] # 表示其他数值型列与 Response 列之间的相关性。去除Response自相关性列（通常为 1）。
df_corr.sort_values()#提取并处理后的相关性值进行排序

#数字特征的分布
#设置 Seaborn 风格
sns.set_palette("Set2")
#选择可视化数值特征
numerical_cols = ['Age', 'Region_Code', 'Annual_Premium', 'Vintage']
#创建 4 行 2 列的子绘图
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
#遍历所选的每一列
for i, column in enumerate(numerical_cols):
    #左侧为无色调图
    sns.histplot(data=train, x=column, kde=True, bins=20, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {column}')
    axes[i, 0].set_xlabel(column)
    axes[i, 0].set_ylabel('Frequency')
    # 在右侧绘制色调图
    sns.boxplot(data=train, y=column, x='Response', ax=axes[i, 1])
    axes[i, 1].set_title(f'Distribution of {column} with Response Hue')
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel('Frequency')
sns.despine()
plt.tight_layout()
plt.show()

#分类特征的分布
# 选择可视化的分类特征
categorical_cols = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']
# 创建 5 行 2 列的子绘图
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
# 遍历所选的每一列
for i, column in enumerate(categorical_cols):
    # 左侧为无色调图
    sns.countplot(x=column, data=train, ax=axes[i, 0])
    axes[i, 0].set_title(f'{column} Count')
    # 在右侧绘制色调图
    sns.countplot(x=column, hue='Response', data=train, ax=axes[i, 1])
    axes[i, 1].set_title(f'{column} Count with Response Hue')
sns.despine()
plt.tight_layout()
plt.show()

#目标变量的分布
# 定义响应类别并计算出现次数
categories = [0, 1]
counts = train.Response.value_counts().tolist()
# 从 Seaborn 中为饼图选择调色板
colors = sns.color_palette("Set2")
# 绘制包含各回答类别计数的饼图
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Response Distribution Pie Chart')
plt.show()

#可视化观察结果
#保单销售渠道与年龄之间存在 -0.59 的负相关。此外，与响应变量之间也没有很强的相关性。与 "响应 "变量相关性最高的是 "曾否投保"，为-0.35。
#年龄变量呈右斜分布。
#年度保费的右尾部较长，表明存在异常值。
#年份变量呈均匀分布。
#老年人的回复率较高。
#数据集中男性略多于女性。
#车辆损坏变量分布均匀，50% 显示损坏，50% 未显示损坏。
#我们的响应变量是不平衡的，87.7% 的实例为 0，只有 12.3% 的实例为 1。

#数据工程和数据预处理
#定义单次编码的分类列
categorical_columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
#直接使用 pandas 进行单次编码
train_encoded = pd.get_dummies(train, columns=categorical_columns, drop_first=True, dtype=int)  

#功能与目标分离
#分离特征 (X) 和目标变量 (y)
X = train_encoded.loc[:, train_encoded.columns != "Response"]
y = train_encoded['Response']

#数据缩放
#使用 StandardScaler 实现功能标准化
scaler = StandardScaler()
#将缩放器与数据相匹配
scaler.fit(X)
#转换数据
X_scaled = scaler.transform(X)

#数据分割 训练/测试分离
#将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)

#机器学习 - XGBoost
#由于响应变量的不平衡，我们将使用 scale_pos_weight 参数来调整类的权重，从而在模型训练过程中平衡数据集。
xgb_params = {    
        'max_depth': 13, 
        'min_child_weight': 5,
        'learning_rate': 0.02,
        'colsample_bytree': 0.6,         
        'max_bin': 3000, 
        'n_estimators': 1500 
}
# 计算负类与正类的比例
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
# 使用指定的超参数初始化 XGBoost 分类器
model = XGBClassifier(**xgb_params, scale_pos_weight=ratio)
# 将分类器与训练数据拟合
XGB_model = model.fit(X_train, y_train)
# 根据测试数据进行预测
predictions = XGB_model.predict_proba(X_test)[:,1]
# 打印曲线下的验证区域
print("Validation Area Under the Curve (AUC): ", roc_auc_score(y_test, predictions))

#提交结果
# 将 "id "列提取到 test_ids 中
test_ids = test['id']
# 在测试文件中删除 ID
test.drop('id', axis=1, inplace=True)
# 直接使用 pandas 进行单次编码
test_encoded = pd.get_dummies(test, columns=categorical_columns, drop_first=True, dtype=int)  
# 使用与 `train` 相同的缩放器转换 `test` 数据
test_scaled = scaler.transform(test_encoded)
# 根据测试数据进行预测
predictions_test = XGB_model.predict_proba(test_scaled)[:,1]
# 结果df
result = pd.DataFrame({'id' : test_ids, 'Response' : predictions_test.flatten()}, 
                      columns=['id', 'Response'])
# 将 df 保存为 csv
result.to_csv("/Users/lulei/Desktop/机器学习/submission保险交叉销售.csv",index=False)