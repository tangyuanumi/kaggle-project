#EDA (探索性数据分析) - 共享单车需求
import numpy as np
import pandas as pd
# 데이터 경로
data_path = 'C:/Users/user/OneDrive/デスクトップ/py/bike-sharing-demand/'
train = pd.read_csv(data_path + 'train.csv') 
test = pd.read_csv(data_path + 'test.csv') 
submission = pd.read_csv(data_path + 'sampleSubmission.csv') 
train.shape, test.shape
train.head()
test.head()
#由于测试数据中没有休闲和注册的特征，因此在训练模型时，我们需要从训练数据中减去休闲和注册的特征。
#这里的标识值（日期时间）只是用来分隔数据，并不能帮助我们预测目标值，因此我们计划在将来训练模型时从训练数据中移除日期时间特征。
#使用 info 函数，我们可以找出 DataFrame 每一列中有多少缺失值，以及数据类型是什么。
train.info()
#训练数据中没有缺失值，因为所有特征的非空计数为 10 886，与数据总数相同。如果存在缺失值，则需要适当处理。
#可以用特征的平均值、中位数和最小值替换缺失值，也可以完全删除包含缺失值的特征。
#或者，也可以将缺失值视为目标，使用其他特征来预测缺失值。您可以将无缺失值的数据视为训练数据，将有缺失值的数据视为测试数据，从而建立模型。
test.info()
#测试数据也没有缺失值，数据类型与训练数据相同。

#特征工程
#完成一些基本分析后，就该将数据可视化了。这是因为从不同角度对数据进行可视化可以揭示原始数据状态下难以发现的趋势、共性和差异。
#有些数据的形式可能不适合可视化。在本竞赛中，日期时间特征就是这种情况。让我们分析一下这个特征，并对其进行转换（特征工程），使其适合可视化。
#日期时间特征的数据类型是对象。在 Pandas 中，对象类型是字符串类型。日期时间由年、月、日、时、分和秒组成，因此我们将其分解为各个组成部分来详细分析。
#使用 Python 内置函数 split() 将日期时间特征拆分为年、月、日、时、分和秒。
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
train.head()
#现在，让我们也创建一个星期功能，它可以通过使用日历和日期时间 "库 "来创建。其中，datetime 是一个用于操作日期和时间的库，与日期时间功能不同。让我们一步步看看如何从日期字符串中提取星期。
#首先，我们需要导入 datetime 库。
import datetime
#然后，我们可以使用 datetime.datetime.strptime() 函数将日期字符串转换为日期时间对象。
date_str = '2021-01-01'
date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
#现在，我们可以使用 date_obj.weekday() 函数来获取日期的星期。星期从 0（星期一）到 6（星期日）计数。
date_obj.weekday()
#因此，我们可以创建一个星期功能，它将日期字符串转换为星期。
train['weekday'] = train['datetime'].apply(lambda x: x.weekday())
train.head()

#接下来是季节和天气特征，它们都是分类数据，但目前用数字 1、2、3 和 4 表示，因此很难知道它们的确切含义。让我们使用 map() 函数把它们转换成字符串，这样在可视化时它们的含义就会更清楚。
train['season'] = train['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
train['weather'] = train['weather'].map({1: 'Clear',
                                         2: 'Mist, Few clouds',
                                         3: 'Light Snow, Rain, Thunderstorm',
                                         4: 'Heavy Rain, Thunderstorm, Snow, Fog'})
train.head()
#增加了日期、年、月、日、时、分、秒和工作日特征，并将季节和天气特征从数字改为字母。将季节和天气特征从数字改为字母。
#请注意，日期特征提供的信息也存在于年、月和日特征中，因此我们将来会删除日期特征。此外，由三个月组成的一组会成为一个季节，这意味着由三个月组成的细粒度月份特征将与季节特征具有相同的含义。
#有时，将过于细化的特征合并到一个更大的分类中会提高性能，因此我们将保留季节特征，删除月份特征。

#数据可视化
#将添加特征的训练数据可视化为图表。可视化是 EDA 最重要的部分。
#它能让我们一目了然地了解数据的分布或数据之间的关系。它还能提供有助于我们建模的信息。
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
#分布图
#分布图是显示数值数据总值的图形。例如，总值是指总计数或百分比。在本章中，我们将绘制 COUNT 的分布图，因为了解目标值的分布将有助于我们知道是按原样使用目标值，还是对其进行转换以进行训练。
mpl.rc('font', size=15)  # 设置字体大小
sns.displot(train['count'])
plt.show()
#x 轴代表目标值，y 轴代表目标值的计数总数。从分布图可以看出，目标值（计数）集中在 0 附近。要使回归模型表现良好，数据必须呈正态分布，而当前的目标值（计数）并不呈正态分布。因此，使用当前目标值建模不可能有好的结果。
#使数据分布更加正态的最常见方法是对数变换。对数变换用于数据向左倾斜的情况，如计数分布。对数变换很简单。只需取您想要的值的对数即可。
sns.displot(np.log(train['count']))
plt.show()
#与转换前相比，它更接近正态分布。我们说过，目标值分布越接近正态分布，回归模型的表现就越好。换句话说，预测 log(count)比直接从特征预测计数更准确。因此，我们要将目标值转换为 log(count)。
#不过，我们需要在最后进行指数转换，以还原实际目标值--计数。如下式所示，将 log(y) 乘以指数就得到了 y。

#条形图
#接下来，我们将绘制一个条形图，显示年、月、日、小时、分钟和秒这六个特征的平均租用次数。这些特征都是分类数据。我们想了解每个分类数据的平均租用次数是如何变化的，从而知道哪些特征是重要的。这就是条形图的作用所在。条形图可以用 seaborn 的 barplot() 函数绘制。
# Step 1 : 为第 m 行第 n 列准备图表
mpl.rc('font', size=14)                       # 设置字体大小
mpl.rc('axes', titlesize=15)                  # 设置每个轴的标题大小
figure, axes = plt.subplots(nrows=3, ncols=2) # 第 3 行 第 2 列 创建图表
plt.tight_layout()                            # 图表间距
figure.set_size_inches(10, 9)                 # 将整个图形尺寸设置为 10x9 英寸

# Step 2 : 为每个轴指定子绘图
##将按年、月、日、时、分、秒分列的平均租金条形图分配到每个轴上
sns.barplot(x='year', y='count', data=train, ax=axes[0, 0])
sns.barplot(x='month', y='count', data=train, ax=axes[0, 1])
sns.barplot(x='day', y='count', data=train, ax=axes[1, 0])
sns.barplot(x='hour', y='count', data=train, ax=axes[1, 1])
sns.barplot(x='minute', y='count', data=train, ax=axes[2, 0])
sns.barplot(x='second', y='count', data=train, ax=axes[2, 1])

# Step 3 : 详细设置
## 3-1 : 给子图起标题
axes[0, 0].set(title='Rental amounts by year')
axes[0, 1].set(title='Rental amounts by month')
axes[1, 0].set(title='Rental amounts by day')
axes[1, 1].set(title='Rental amounts by hour')
axes[2, 0].set(title='Rental amounts by minute')
axes[2, 1].set(title='Rental amounts by second')

## 3-2 : 将第2行子图的 X 轴标签旋转 90 度
axes[1, 0].tick_params(axis='x', labelrotation=90)
axes[1, 1].tick_params(axis='x', labelrotation=90)
plt.show()
#分析结果
#1. 按年份分列的平均租金"：2012 年的租金高于 2011 年。
#2. 按月平均出租率"：6 月份的平均出租率最高，1 月份最低。可以猜测，天气越暖和，租房人数越多。
#3. 日均租赁量"：日均租赁量没有明显差异。正如我在介绍页面所说，训练数据只有每月 1 日至 19 日的数据。
#其余从每月 20 日到月底的数据都是测试数据。因此，我们不能使用 "日 "作为特征，因为训练数据中的 "日 "和测试数据中的 "日 "具有完全不同的值。
#4. 平均每小时租用次数"：图表形状为钟形曲线。早上 4 点的租车次数最少，这很合理，因为早上 4 点很少有人骑车。另一方面，早上 8 点和下午 5-6 点的租车次数最多。
#5. 平均每分每秒租赁次数"：它不包含任何信息，因此我们在以后训练模型时不会使用分和秒的特征。

#箱形图
#方框图是一种基于分类数据来表示数值数据信息的图表。它的特点是比条形图提供更多的信息。
#在本例中，我们将按季节、天气、节假日和工作日（分类数据）对租赁数量（数值数据）进行方框图绘制。
#您可以看到我们的目标值，即租赁次数，是如何根据每个分类数据发生变化的。
# 步骤 1：为第 m 行第 n 列准备图表
figure, axes = plt.subplots(nrows=2, ncols=2) #  2 行 2 列
plt.tight_layout()
figure.set_size_inches(10, 13)

# 步骤 2：分配分计划
## 按季节、天气、节假日和工作日分列的租赁量方框图
sns.boxplot(x='season', y='count', data=train, ax=axes[0, 0])
sns.boxplot(x='weather', y='count', data=train, ax=axes[0, 1])
sns.boxplot(x='holiday', y='count', data=train, ax=axes[1, 0])
sns.boxplot(x='workingday', y='count', data=train, ax=axes[1, 1])

## 步骤 3：详细设置
## 3-1 : 给分图起标题
axes[0, 0].set(title='Box Plot On Count Across Season')
axes[0, 1].set(title='Box Plot On Count Across Weather')
axes[1, 0].set(title='Box Plot On Count Across Holiday')
axes[1, 1].set(title='Box Plot On Count Across Working Day')

## 3-2 : 解决轴标签重叠问题
axes[0, 1].tick_params(axis='x', labelrotation=10) # 旋转 10 度
plt.show()
#春季自行车租赁数量最少，秋季最多。
#方框图显示了不同天气的租赁数量，这与我们的直觉相符：天气好的时候，租赁数量最多，天气不好的时候，租赁数量很少。大雨和大雪时几乎没有租车（图中最右边的方框）。
#方框图显示了根据是否有公共假日计算的租房数量。x 轴标注 0 表示不放假，1 表示放假。节假日和非节假日的自行车租赁次数中位数几乎相同。不过，在非节假日有很多异常值。
#显示了根据是否为工作日计算的租赁数量，工作日的异常值更多。请注意，工作日是指除节假日和周末以外的任何一天。

#点阵图
#我们把工作日、节假日、平日、季节和天气的每小时平均租车次数绘制成点图。点图以点和线的形式显示基于分类数据的数值数据的平均值和置信区间。
#它提供的信息与条形图相同，但更适合在一个屏幕上绘制多个图形，以便相互比较。
# 步骤 1：准备第 m 行和第 n 列数字
mpl.rc('font', size = 11)
figure, axes = plt. subplots(nrows = 5) #  5 行 1 列
figure.set_size_inches(12, 18)

# 步骤 2：分配分计划
#根据工作日、节假日、一周中的哪几天、季节和天气，按时间点绘制平均租赁量图
sns.pointplot(x = 'hour', y = 'count', data = train, hue = 'workingday', ax = axes[0])
sns.pointplot(x = 'hour', y = 'count', data = train, hue = 'holiday', ax = axes[1])
sns.pointplot(x = 'hour', y = 'count', data = train, hue = 'weekday', ax = axes[2])
sns.pointplot(x = 'hour', y = 'count', data = train, hue = 'season', ax = axes[3])
sns.pointplot(x = 'hour', y = 'count', data = train, hue = 'weather', ax = axes[4])
plt.show()
#分析结果
#在工作日，高峰时段的租车次数最多，而在休息日，中午 12 点至下午 2 点的租车次数最多。
#按星期（无论是否节假日）划分的点图与按工作日划分的点图相似。
#从按季节和时间划分的点图来看，秋季的租用次数最多，春季最少。
#不出所料，天气好的时候租用次数最多，但在大雨和大雪时，18:00 有少数租用次数。 您可以考虑删除这些异常值。

#带有回归线的散点图
#让我们把温度、风速、湿度等数值数据按租借量绘制成 "带回归线的散点图"。带回归线的散点图用于确定数值数据之间的相关性。
#该图可以用 seaborn 的 regplot() 函数绘制。
#步骤 1：准备一个 m 型 n 色谱柱图
mpl.rc('font', size = 15)
figure, axes = plt.subplots(nrows = 2, ncols = 2) # 2행 2열
plt.tight_layout()
figure.set_size_inches(7, 6)

# 步骤 2：分配分计划
## 按温度、风速、风寒和湿度绘制的租赁量散点图
sns.regplot(x = 'temp', y = 'count', data = train, ax = axes[0, 0],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
sns.regplot(x = 'atemp', y = 'count', data = train, ax = axes[0, 1],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
sns.regplot(x = 'windspeed', y = 'count', data = train, ax = axes[1, 0],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
sns.regplot(x = 'humidity', y = 'count', data = train, ax = axes[1, 1],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
plt.show()
#分析结果
#通过回归线的斜率可以大致了解趋势。
#温度越高，租金越高。湿度越低，租金越高。换句话说，温暖的时候比寒冷的时候有更多的出租，不潮湿的时候比潮湿的时候有更多的出租。
#回归线显示，风速越大，租用次数越多。这有点奇怪，因为我以为风越小，租房人数越多。原因是风速特征中有很多缺失值。
#如果仔细观察，有很多数据的风速为 0。很可能实际风速并不是 0，而是由于没有观察到或出现错误而记录为 0。
#缺失值的数量使得很难根据图表将风速与租房次数联系起来。由于缺失值较多，很难根据图表将风速与租金数量联系起来。应适当处理缺失值较多的数据：用其他值替换缺失值，或删除风速特征本身。

#热图 (Heatmap)
#温度、温度系数、湿度、风速和计数都是数值数据。让我们看看它们是如何相关联的。
# corr() 函数计算 DataFrame 中特征之间的相关系数，并返回 "数值数据之间的相关矩阵"。
#面对如此多的组合，我们很难一眼看出哪些特征是紧密相关的。这就是热图的作用所在。
#热图是数据之间关系的彩色表示，可以让人一目了然地比较多个数据。热图可以使用 seaborn() 中的 heatmap() 函数绘制。
#特征之间的相关矩阵
corrMat = train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
sns.heatmap(corrMat, annot = True) # 绘制相关热图
ax.set(title = "Heatmap of Numerical Data")
plt.show()
#分析结果
#温度和温度与租用次数（计数）之间的相关系数为 0.39。这是一种正相关关系，即温度越高，租用次数越多。
#另一方面，湿度与租用次数之间呈负相关，即湿度越 "低"，租用次数越多。这与我们之前在散点图中分析的结果相同。
#风速和租金之间的相关性为 0.1。由于相关性很弱，我们将删除风速特征。

#分析摘要和建模战略
#**1. 转换目标值
#检查分布后，我们发现目标值（即计数）在零附近偏斜，因此我们需要对其进行对数化处理，使其更符合正态分布。
#我们将目标值转换为 log（count）而不是 count，并在最后将其指数化为 count。
#**2. 添加派生特征
#由于日期时间特征是不同信息的混合体，我们可以将它们分开，创建年、月、日、时、分和秒特征。
#我们将添加工作日特征，这是隐藏在日期时间特征中的另一条信息。
#3. 删除衍生特征
#使用测试数据中没有的特征进行训练是没有意义的，因此我们要删除只存在于训练数据中的随意特征和注册特征。
#日期时间特征只是一个索引，因此对我们预测目标值没有任何帮助。
#日期特征提供的信息包含在新添加的年、月和日特征中，因此我们将删除它们。
#月份特征可以看作是季节特征的一个分支。如果数据过于精细，每次分类的数据量就会减少，这实际上会阻碍学习。
#检查条形图，移除衍生特征 "日"，因为它不具有区分度，而 "分 "和 "秒 "则不包含任何信息。
#检查散点图和热图后，发现风速特征有很多缺失值，而且与租房次数的相关性很弱。
#**4. 去除异常值
#检查点图，天气 = 4 的数据是一个异常值。

#建模策略
#如果想在比赛中取得好成绩，就需要建立自己的优化模型。
#基准模型：采用最基本的线性回归模型
#性能改进： Ridge、Lasseau 和随机森林回归模型
#特征工程：对所有模型进行与前一级相同的分析
#超参数优化： 网格搜索
#杂项： 目标值为对数（计数），而不是计数
