#RFM(模型)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Pie
import pyecharts.options as opts
from matplotlib.ticker import FuncFormatter
import plotly.graph_objs as go
import plotly as py

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data1.csv', encoding='utf-8')
# df.head(3)
# 统计缺失率
df.apply(lambda x: sum(x.isnull()) / len(x), axis=0)
df.drop(['Description'], axis=1, inplace=True) # Description字段对该数据分析目标无意义，删除
df['CustomerID'] = df['CustomerID'].fillna('U') # 缺失的用户ID填充为'U'
df['amount'] = df['Quantity'] * df['UnitPrice'] # 每个订单的发生额

# 将 InvoiceDate字段拆分成两列，之后删除InvoiceDate字段
'''plan A:'''
# df['Date'] = [i.split(' ')[0] for i in df['InvoiceDate']]
# df['Time'] = [i.split(' ')[1] for i in df['InvoiceDate']]

'''plan B:'''
# df = df.join(df['InvoiceDate'].str.split(' ', expand=True))
# df.columns = df.columns.tolist()[:-2] + ['Date', 'Time']

'''plan C'''
df['Date'] = pd.to_datetime(df['InvoiceDate']).dt.date
df['Time'] = pd.to_datetime(df['InvoiceDate']).dt.time
df['Date'] = pd.to_datetime(df['Date'])

df.drop(['InvoiceDate'], axis=1, inplace=True)
df = df.drop_duplicates() # 删除重复值


df2 = df.loc[df['UnitPrice'] <= 0]  # 对单价进行异常分析
df2.shape[0]/df.shape[0]
# 异常值中单价的分类
# df2['UnitPrice'].groupby(df2['UnitPrice']).count()
df2['UnitPrice'].value_counts()

df1 = df.loc[df['Quantity'] <= 0]

returns = pd.pivot_table(df1, index=df1['Date'].dt.year, columns=df1['Date'].dt.month, values='amount', aggfunc={'amount':np.sum})
returns # 退货情况
df3 = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)] # 筛选出销售的正常数据

sales = pd.pivot_table(df3, index=df3['Date'].dt.year, columns=df3['Date'].dt.month, values='amount', aggfunc={'amount':np.sum})
sales # 营业额
return_rate = np.abs(returns) / sales # 退货率
return_rate

avg_return = return_rate[1:2].mean(axis=1).values[0]
avg_return  # 平均退货率

# 取出2020年退货率数据，用于画图
return_rate_11 = [round(i, 2) for i in return_rate.values.tolist()[1]]
month = return_rate.columns.tolist()

plt.figure(figsize=(10,5), dpi=70)
plt.style.use('fivethirtyeight')
plt.plot(month, return_rate_11, 'bo-', lw=2, label='月退货率')
plt.title('每月退货率', fontdict={'color':'black', 'fontsize':16}, pad=12)
plt.xlabel('月份', fontdict={'color':'black', 'fontsize':14})
plt.ylabel('退货率', fontdict={'color':'black', 'fontsize':14})
# plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
plt.yticks([])
plt.xticks(np.arange(1, 13))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, position: '{:.1f}%'.format(x*100)))
for i, j in zip(month, return_rate_11):
    plt.text(i-0.02, j-0.02, '{:.1f}%'.format(j*100), bbox=dict(facecolor='red', alpha=0.1))
plt.axhline(y=avg_return,ls='--', color='r', lw=2, label='平均退货率')
plt.annotate('{:.1f}%'.format(round(avg_return, 3)*100), xy=(8, avg_return), xytext=(8.5, avg_return+0.05), arrowprops=dict(facecolor='red', shrink=0.05))
plt.grid(b=False)
plt.legend(loc='best')
plt.show()


customer_newest_consume = df3.groupby('CustomerID')['Date'].max() # 每位用户最近一次购买时间
customer_newest_consume.head()
newest_time_consume = df3['Date'].max()  # 目标时间（最近一次购买时间）
value_R = (newest_time_consume - customer_newest_consume).dt.days
value_R # 每位用户最近购买时间于目标时间的距离
value_F = df3.groupby('CustomerID')['InvoiceNo'].nunique() # nunique()去重
value_F
value_M = df3.groupby('CustomerID')['amount'].sum()
value_M
value_R.describe()
plt.hist(value_R, bins=30)
plt.show()
value_M.describe()
plt.hist(value_M, bins=30) # 异常值严重影响了数据的分布
plt.show()
plt.hist(value_M[value_M < 2000], bins=30) # 绘制金额小于2000的
plt.show()
value_F.describe() # 中位数是2，而最大值是1428，异常值很严重
plt.hist(value_F[value_F < 20], bins=30)
plt.show()

# 可根据分位数来分段
print(' value_R:\t', value_R.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).tolist())
print(' value_F:\t', value_F.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).tolist())
print(' value_M:\t', [round(i) for i in value_M.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).tolist()])

value_R: [0.0, 5.0, 12.0, 22.0, 32.0, 50.0, 71.0, 108.0, 179.0, 262.2000000000003, 373.0]
value_F: [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 6.0, 9.0, 1428.0]
value_M: [4, 155, 250, 350, 488, 669, 934, 1348, 2056, 3641, 1754902]
# 构建分段指标
bins_R = [0, 30, 90, 180, 360, 720]
bins_F = [1, 2, 5, 10, 20, 5000]
bins_M = [0, 500, 2000, 5000, 10000, 200000]
# 分段，label可以理解为权重，对R而言，值越小表示里目标时间近，所占权重也就更大，F,M同理
score_R = pd.cut(value_R, bins_R, labels=[5,4,3,2,1], right=False)
score_F = pd.cut(value_F, bins_F, labels=[1,2,3,4,5], right=False)
score_M = pd.cut(value_M, bins_M, labels=[1,2,3,4,5], right=False)

rfm = pd.concat([score_R, score_F, score_M], axis=1)
rfm.rename(columns={'Date':'R_value', 'InvoiceNo':'F_value', 'amount':'M_value'}, inplace=True)
# 转换类型用于计算
for i in ['R_value', 'F_value', 'M_value']:
    rfm[i] = rfm[i].astype(float)
rfm.describe()

rfm['R'] = np.where(rfm['R_value'] > 3.82, '高', '低')
rfm['F'] = np.where(rfm['F_value'] > 2.03, '高', '低')
rfm['M'] = np.where(rfm['M_value'] > 1.89, '高', '低')

rfm['value'] = rfm['R'].str[:] + rfm['F'].str[:] + rfm['M'].str[:]
rfm['value'] = rfm['value'].str.strip() # 去除空格处理
def trans_value(x):
    if x == '高高高': return '重要价值客户'
    elif x == '高低高': return '重要发展客户'
    elif x == '高高低': return '一般价值客户'
    elif x == '低高高': return '重要保持客户'
    elif x == '低低高': return '重要挽留客户'
    elif x == '高低低': return '一般发展客户'
    elif x == '低高低': return '一般保持客户'
    else: return '一般挽留客户'
rfm['用户等级'] = rfm['value'].apply(trans_value)

re = rfm['用户等级'].value_counts()
plt.figure(figsize=(9,5))
plt.bar(re.index.tolist(), re.values.tolist(), width=0.5)
plt.grid(b=False)
plt.xticks(rotation=45)
plt.title('用户等级')
plt.show()

trace = [go.Pie(labels=re.index, values=re.values,textfont=dict(size=12,color='white'))]
layoyt = go.Layout(title='用户等级比例')
figure_basic1 = go.Figure(data=trace, layout=layoyt)
py.offline.plot(figure_basic1,filename='用户等级比例.html')





