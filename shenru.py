#RFM(模型)

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly as py
import plotly.graph_objs as go
import seaborn as sns
from matplotlib.ticker import FuncFormatter



warnings.filterwarnings('ignore')
os.chdir(r'G:\code\shuji\shenru')
pyplot = py.offline.iplot
py.offline.init_notebook_mode()

df = pd.read_csv('data1.csv',encoding='utf-8', dtype={'CustomerID': str})

#数据清洗
df.apply(lambda x: sum(x.isnull())/len(x), axis=0)    # 统计缺失率
df.drop(['Description'], axis=1, inplace=True)    # Description字段对该数据分析目标无意义，删除
df['CustomerID'] = df['CustomerID'].fillna('U')   # 缺失的用户ID填充为'U'
df['amount'] = df['Quantity'] * df['UnitPrice']   # 每个订单的发生额
df['date'] = [x.split(' ')[0] for x in df['InvoiceDate']]#拆分时间InvoiceDate
df['time'] = [x.split(' ')[1] for x in df['InvoiceDate']]
df.drop(['InvoiceDate'], axis=1, inplace=True)

df['year'] = [x.split('/')[2] for x in df['date']] #拆分年月日
df['month'] = [x.split('/')[0] for x in df['date']]
df['day'] = [x.split('/')[1] for x in df['date']]

df['date'] = pd.to_datetime(df['date'])
df = df.drop_duplicates()#去重
# 对单价进行异常分析
df2 = df.loc[df['UnitPrice']<0]
# 异常值中单价的分类
# df2['UnitPrice'].value_counts()
df2['UnitPrice'].groupby(df2['UnitPrice']).count()

df1 = df.loc[df['Quantity'] <= 0]


returns = pd.pivot_table(df1,index='year',columns='month',values='amount',aggfunc={'amount':np.sum}) # 退货情况
df2 = df[(df['Quantity']>0) & (df['UnitPrice'] > 0)] #退货 & 赠品系统错误
sales = pd.pivot_table(df2,index='year',columns='month',values='amount',aggfunc={'amount':np.sum}) # 营业额

return_rate = np.abs(returns) / sales #退货率
avg_return = return_rate[1:2].mean(axis=1).values[0]  # 平均退货率

# 取出2020年退货率数据，用于画图
# return_rate_11 = [round(i, 2) for i in return_rate.values.tolist()[1]]
# month = return_rate.columns.tolist()
#
# plt.figure(figsize=(10,5), dpi=70)
# plt.style.use('fivethirtyeight')
# plt.plot(month, return_rate_11, 'bo-', lw=2, label='月退货率')
# plt.title('每月退货率', fontdict={'color':'black', 'fontsize':16}, pad=12)
# plt.xlabel('月份', fontdict={'color':'black', 'fontsize':14})
# plt.ylabel('退货率', fontdict={'color':'black', 'fontsize':14})
# # plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
# plt.yticks([])
# plt.xticks(np.arange(1, 13))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, position: '{:.1f}%'.format(x*100)))
# for i, j in zip(month, return_rate_11):
#     plt.text(i-0.02, j-0.02, '{:.1f}%'.format(j*100), bbox=dict(facecolor='red', alpha=0.1))
# plt.axhline(y=avg_return,ls='--', color='r', lw=2, label='平均退货率')
# plt.annotate('{:.1f}%'.format(round(avg_return, 3)*100), xy=(8, avg_return), xytext=(8.5, avg_return+0.05), arrowprops=dict(facecolor='red', shrink=0.05))
# plt.grid(b=False)
# plt.legend(loc='best')
# plt.show()


ddd = np.abs(returns/sales).loc['2020'].mean()

R_value = df2.groupby('CustomerID')['date'].max()  # 每位用户最近一次购买时间
R_value = (df2['date'].max() - R_value).dt.days  # 每位用户最近购买时间于目标时间的距离
F_value = df2.groupby('CustomerID')['InvoiceNo'].nunique() #去重
M_value = df2.groupby('CustomerID')['amount'].sum()


#直方图
# sns.set(style='darkgrid')
# plt.hist(M_value[M_value<2000],bins=30) #2000以下# 异常值严重影响了数据的分布
# plt.show()
# 可根据分位数来分段
print(' value_R:\t', R_value.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).tolist())
print(' value_F:\t', F_value.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).tolist())
print(' value_M:\t', [round(i) for i in M_value.quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).tolist()])


R_bins = [0, 30, 90, 180, 360, 720]     # 构建分段指标
F_bins = [1, 2, 5, 10, 20, 5000]
M_bins = [0, 500, 2000, 5000, 10000, 2000000]
R_score = pd.cut(R_value, R_bins, labels=[5,4,3,2,1,], right=False) #间隔  # 分段，label可以理解为权重，对R而言，值越小表示里目标时间近，所占权重也就更大，F,M同理
F_score = pd.cut(F_value, F_bins, labels=[1,2,3,4,5], right=False) #消费频数
M_score = pd.cut(M_value, M_bins, labels=[1,2,3,4,5], right=False) #消费金额
rfm = pd.concat([R_score, F_score, M_score], axis=1)  # 横向合并，axis=1；纵向合并，axis=0(默认)
rfm.rename(columns={'date':'R_score','InvoiceNo':'F_score','amount':'M_score'},inplace=True)


for i in ['R_score','F_score','M_score']: # 转换类型用于计算
    rfm[i] = rfm[i].astype(float)

rfm['R'] = np.where(rfm['R_score']> 3.82, '高', '低') # 根据平均值构建分级
rfm['F'] = np.where(rfm['F_score']> 2.03, '高', '低')
rfm['M'] = np.where(rfm['M_score']> 1.89, '高', '低')
rfm['value'] = rfm['R'].str[:] + rfm['F'].str[:]+ rfm['M'].str[:]

rfm['value'] = rfm['value'].str.strip()
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
#可视化
trace_basic = [go.Bar(x=re.index, y=re.values,marker=dict(color='orange'), opacity=0.50)]
layout = go.Layout(title='用户等级情况',xaxis=dict(title= '用户重要程度'))
figure_basic = go.Figure(data=trace_basic, layout=layout)
py.offline.plot(figure_basic,filename='用户情况.html')



trace = [go.Pie(labels=re.index, values=re.values,textfont=dict(size=12,color='white'))]
layoyt = go.Layout(title='用户等级比例')
figure_basic1 = go.Figure(data=trace, layout=layoyt)
py.offline.plot(figure_basic1,filename='用户等级比例.html')

# pd.set_option('display.max_columns', None)#显示全部结果
# pd.set_option('display.max_rows', None)
# print(rfm)








