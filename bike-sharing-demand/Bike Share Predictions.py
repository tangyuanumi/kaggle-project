#自行车共享EDA＆预测
#对 2011 年和 2012 年共享单车需求的日期、时间、温度和风速等各方面的总借出量之间的关系进行了图表和探索性数据分析。
#通过机器学习的随机森林回归模型和深度学习回归模型对测试值进行预测，并将两个数据合并生成新的预测值，共得到三个预测值，并对准确率进行比较。
# Data
import pandas as pd
import numpy as np
import missingno as msno

# Visuallization
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import tensorflow as tf
from tensorflow import compat

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
train = pd.read_csv(r"C:\Users\user\OneDrive\デスクトップ\py\bike-sharing-demand\train.csv")
test = pd.read_csv(r"C:\Users\user\OneDrive\デスクトップ\py\bike-sharing-demand\test.csv")
train.head()
test.head()

#检查缺失值
msno.matrix(train)
plt.show()
msno.matrix(test)
plt.show()
#可视化
train.head()


year = []
month = []
day = []
time = []
for i in range(len(train)):
    year.append(int(train["datetime"].values[i].split()[0].split("-")[0]))
    month.append(int(train["datetime"].values[i].split()[0].split("-")[1]))
    day.append(int(train["datetime"].values[i].split()[0].split("-")[2]))
    time.append(int(train["datetime"].values[i].split()[1].split(":")[0]))
train["year"] = year
train["month"] = month
train["day"] = day
train["time"] = time
train.head()

year = []
month = []
day = []
time = []
for i in range(len(test)):
    year.append(int(test["datetime"].values[i].split()[0].split("-")[0]))
    month.append(int(test["datetime"].values[i].split()[0].split("-")[1]))
    day.append(int(test["datetime"].values[i].split()[0].split("-")[2]))
    time.append(int(test["datetime"].values[i].split()[1].split(":")[0]))
test["year"] = year
test["month"] = month
test["day"] = day
test["time"] = time
test.head()

features = ["season","holiday","workingday","weather","temp","atemp","humidity","windspeed","year","month","day","time"]
x_train = train[features]
x_test = test[features]
y_train = train["count"]

#深度学习 - 回归模型
#深度学习回归模型是一种人工神经网络，用于预测连续值，如股票价格、住房价格或温度。
# 它们由多层相互连接的节点组成，处理输入数据，并根据学习到的输入和输出之间的关系进行预测。
# 这些模型通过梯度下降优化技术对大量数据进行训练，从而调整网络权重，使预测误差最小化。
# 深度学习回归模型的优势在于能够从数据中学习复杂的非线性关系，使其成为各种应用的强大工具。
#+我应该预测共享需求的数量。因此，使用回归模型更有助于获得更准确的预测结果。

#创建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation="linear"))
#使用均方误差损失函数和 L2 正则化编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#在训练数据上训练模型
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train, epochs=200, verbose = 0)
#预测测试数据的 y 值
y_pred = model.predict(x_test)
print("Predicted y values:", y_pred)
#评估模型在测试数据上的性能
#test_loss = model.evaluate(x_test)
#print("Test loss:", test_loss)
deep_learning_prediction = np.array(y_pred)
deep_learning_prediction

#MachineLearning
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
k_fold = KFold(n_splits = 10, shuffle = True , random_state = 0)#创建一个KFold对象，用于实现10折交叉验证。
#通过将数据集分成10个子集，并在每次验证时使用不同的子集作为验证集，其余子集作为训练集，可以更全面地评估模型的性能。

#XGBoost、Catboost、RandomForest 调节器默认分数比较
cbr = CatBoostRegressor(verbose=0)
xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
rfr = RandomForestRegressor()

model_list = [cbr,xgbr,rfr]
model_score = []
for model in model_list:
    model.fit(x_train,y_train)
    model_score.append(model.score(x_train,y_train))
model_score
#比较不同回归模型在同一数据集上的表现，从而选择最适合的模型进行后续的预测任务。
#绘制柱状图，用于比较三种不同机器学习模型（Catboost、XGboost、RandomForest）在同一数据集上的表现。
model_name = ["Catboost","XGboost","RandomForest"]
plt.figure(figsize=(20,8))
x = model_name
y = model_score
for i in range(len(x)):
    height = y[i]
    plt.text(x[i], height, '%.10f' %height, ha='center', va='bottom', size = 12)
plt.ylim(0.96,1)
plt.bar(x,y,color='#5F00FF')
plt.show()
#RandomForestRegressor 在这些数据上表现出更好的性能。

#随机森林回归器
#评估不同n_estimators值对随机森林回归模型性能的影响。
from sklearn.ensemble import RandomForestRegressor
# 根据 "n_estimators "检查结果。
for i in (10, 20, 30, 40, 100, 150,300): #遍历不同的n_estimators值,n_estimators表示随机森林中决策树的数量。
    model = RandomForestRegressor(n_estimators= i,n_jobs= -1, random_state = 15)
    model.fit(x_train,y_train) #对于每个n_estimators值，创建并训练一个随机森林回归模型。
    # n_estimators "值越高，精确度就越高、 
    # 但计算机计算和得出结果所需的运算量也越大。
    # 我将该值适当设置为 300。

    relation_square = model.score(x_train, y_train)#计算模型在训练数据上的R²得分，R²得分用于衡量模型解释目标变量变异的能力。
    print('relation_square : ', relation_square)#打印当前n_estimators值对应的R²得分。
    #可视化预测结果
    plt.figure(figsize=(20,5))#设置图形的大小。
    y_p = model.predict(x_train)#使用训练数据进行预测。
    ax1 = sns.kdeplot(y_train,label = 'y_train',color="red")#绘制训练数据真实值的核密度估计图。
    ax2 = sns.kdeplot(y_p,label = 'y_pred',color="blue")#绘制预测数据的核密度估计图。
    
    plt.title(i)#设置图形的标题为当前的n_estimators值。
    plt.legend()#显示图例。
    plt.show()
#得出n_estimators参数对模型性能的影响，并选择合适的参数值以优化模型。
#R²得分越大，表示模型对目标变量的解释能力越强，模型的拟合程度越好。
#n_estimators=300的R²得分最大，因此我将其作为最终的模型参数。

model = RandomForestRegressor(n_estimators=300, n_jobs = -1 , random_state = 0)
model.fit(x_train, y_train)

predictions = model.predict(x_test)#使用训练好的模型对测试数据x_test进行预测。
predictions
machine_learning_prediction = predictions #机器学习预测结果存储
machine_learning_prediction

deep_learning_prediction1 = [] #用于存储深度学习模型的预测结果。
for i in range(len(deep_learning_prediction)):
    deep_learning_prediction1.append(deep_learning_prediction[i][0])
deep_learning_prediction1 = np.array(deep_learning_prediction1)#转换为 NumPy 数组
deep_learning_prediction1

#绘制核密度估计图，用于比较三种不同模型在同一数据集上的预测结果。
plt.figure(figsize=(20,5))
ax1 = sns.kdeplot(y_train, label = 'train',color="red")
ax2 = sns.kdeplot(machine_learning_prediction, label = 'test-MA',color="blue")
ax3 = sns.kdeplot(deep_learning_prediction1, label = 'test-DP',color="green")
plt.legend()
plt.show()

#合并预测值
#通过上图，机器学习预测和深度学习预测的图形经常相互交叉。它的预测结果具有相反的模式。
#为了弥补这一缺陷，我们尝试通过获取两个值的平均值来得出一个新的预测值。
print(len(deep_learning_prediction))
print(len(machine_learning_prediction))
print(len(y_train))

i = 100
print(deep_learning_prediction[i][0],machine_learning_prediction[i],y_train[i])
#打印第 100 个样本的深度学习模型预测值、机器学习模型预测值以及训练数据中的真实值。
#可以直观地比较不同模型的预测结果与真实值之间的差异，从而评估模型的性能。

import math
final_pred = []
for i in range(6493):
    final_pred.append((deep_learning_prediction[i][0]+machine_learning_prediction[i])/2)
plt.figure(figsize=(20,5)) #将两模型预测值相加后取平均值
ax1 = sns.kdeplot(y_train, label = 'train',color="red")
ax2 = sns.kdeplot(final_pred, label = 'test-MERGE',color="blue")
plt.legend() #绘制核密度估计图以比较训练数据的真实值和合并预测值的分布情况。
plt.show()
#直观地评估合并预测值与真实值的接近程度，从而判断合并预测方法的有效性。

submission = pd.read_csv(r"C:\Users\user\OneDrive\デスクトップ\py\bike-sharing-demand\sampleSubmission.csv")
#总共获得了三种预测结果深度学习预测、机器学习预测、以及两种预测的平均预测结果。
submission["count"] = final_pred
# In this session, I use merge prediction which is the average predict between ML and DL.
# If you want submit machine learning prediction, then use "machine_learning_prediction"
# or if you want submit deep learning prediction, then use "deep_learning_prediction"
submission

submission.to_csv("submission.csv",index=False)
submission = pd.read_csv("submission.csv")
submission.head()

import os
print(os.getcwd())
   
Score_result = pd.DataFrame()
pred_type = ["Merge Pred","Machine Pred","Deep Pred"]
pred_score = [0.46969,0.48616,0.55899]
Score_result.insert(0,"Pred_Type",pred_type)
Score_result.insert(0,"Pred_Score",pred_score)
# Score_result.set_index("Pred_Type",pred_type)
# Score_result["Pred_Score"]
plt.figure(figsize=(10,10))
plt.title("Score Comparison Graph")
plt.show()
sns.barplot(data = Score_result, x = Score_result["Pred_Type"],y = Score_result["Pred_Score"])