import plotly as py

py.offline.init_notebook_mode()
pyplot = py.offline.iplot

import plotly.graph_objs as go
from plotly.graph_objs import Scatter

from scipy import stats

import pandas as pd

import numpy as np

import seaborn as sns

sns.set(style='darkgrid')

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import os

data = pd.read_csv('LCIS.csv')

# 修改列名
data.rename(columns={'ListingId': '列表序号', 'recorddate': '记录日期'}, inplace=True)

# 缺失率
miss_rate = pd.DataFrame(data.apply(lambda x: sum(x.isnull()) / len(x)))

# 将缺失率保存为一列
miss_rate.columns = ['缺失率']

# 缺失率以三位小数百分数形式表示
miss_rate[miss_rate['缺失率'] > 0]['缺失率'].apply(lambda x: format(x, '.3%'))

# 计数‘下次计划还款利息’为缺失值(已还清)的用户的‘标当前状态’
data[data['下次计划还款利息'].isnull()]['标当前状态'].value_counts()

# 显示'上次还款利息'为缺失值的用户信息的后九列
data[data['上次还款利息'].isnull()].iloc[:, -9:-1]

# 查看历史成功借款金额缺失的用户情况
data[data['历史成功借款金额'].isnull()]

# 查看记录日期缺失的用户情况
data[data['记录日期'].isnull()][['手机认证', '户口认证']]

# 删除记录日期缺失的用户数据
data.dropna(subset=['记录日期'], how='any', inplace=True)

# 去重画图
data[data.duplicated()]
data['手机认证'].value_counts().plot(kind='bar')

# 取出’手机认证’一列中的'成功认证'和'未成功认证'，其他删除
data = data[(data['手机认证'] == '成功认证') | (data['手机认证'] == '未成功认证')]
data = data[(data['户口认证'] == '成功认证') | (data['户口认证'] == '未成功认证')]
data = data[(data['视频认证'] == '成功认证') | (data['视频认证'] == '未成功认证')]
data = data[(data['学历认证'] == '成功认证') | (data['学历认证'] == '未成功认证')]
data = data[(data['征信认证'] == '成功认证') | (data['征信认证'] == '未成功认证')]
data = data[(data['淘宝认证'] == '成功认证') | (data['淘宝认证'] == '未成功认证')]

# 不同性别的放贷比例与逾期关系
df_gender = pd.pivot_table(data=data, columns='标当前状态', index='性别', values='列表序号', aggfunc=np.size)

# 借款笔数占比
df_gender['借款笔数占比'] = df_gender.apply(np.sum, axis=1) / df_gender.sum().sum()

# 逾期笔数占比
df_gender['逾期笔数占比'] = df_gender['逾期中'] / df_gender.sum(axis=1)

# 画图
plt.figure(figsize=(16, 9))

plt.subplot(121)
plt.bar(x=df_gender.index, height=df_gender['借款笔数占比'], color=['c', 'g'])
plt.title('男女借款比例')

plt.subplot(122)
plt.bar(x=df_gender.index, height=df_gender['逾期笔数占比'], color=['c', 'g'])
plt.title('男女逾期情况')

plt.suptitle('不同性别的客户画像')
plt.show()

# 借款累计金额占比
df_age = data.groupby(['年龄'])['借款金额'].sum()
df_age = pd.DataFrame(df_age)
df_age['借款金额累计'] = df_age['借款金额'].cumsum()
df_age['借款累计金额占比'] = df_age['借款金额累计'] / df_age['借款金额'].sum()
df_age

# 80%的贷款借给了36岁以下的用户
index_num = df_age[df_age['借款累计金额占比'] > 0.8].index[0]

# 画图
cum_percent = df_age.loc[index_num, '借款累计金额占比']
plt.figure(figsize=(16, 9))
plt.bar(x=df_age.index, height=df_age['借款金额'], color='steelblue', alpha=0.5, width=0.7)
plt.xlabel('年龄', fontsize=20)
plt.axvline(x=index_num, color='orange', linestyle='--', alpha=0.8)
df_age['借款累计金额占比'].plot(style='--ob', secondary_y=True)
plt.text(index_num + 0.4, cum_percent, '累计占比为:%.3f%%' % (cum_percent * 100), color='indianred')
plt.show()

# 年龄与借款的情况
# 按照年龄分层
data['age_bin'] = pd.cut(data['年龄'], [17, 24, 30, 36, 42, 48, 54, 65], right=True)
# 查看每个年龄段的情况
df_age = pd.pivot_table(data=data, columns='标当前状态', index='age_bin', values='列表序号', aggfunc=np.size)
# 总的借款笔数
df_age['借款笔数'] = df_age.sum(axis=1)
# 借款笔数分布
df_age['借款笔数分布'] = df_age['借款笔数'] / df_age['借款笔数'].sum()
# 逾期占比
df_age['逾期占比'] = df_age['逾期中'] / df_age['借款笔数']
# 变为百分数形式
df_age['借款笔数分布%'] = df_age['借款笔数分布'].apply(lambda x: format(x, '.3%'))
df_age['逾期占比%'] = df_age['逾期占比'].apply(lambda x: format(x, '.3%'))

# 画图
plt.figure(figsize=(16, 9))
df_age['借款笔数分布'].plot(kind='bar', rot=45, color='steelblue', alpha=0.5)
plt.xlabel('年龄分段情况')
plt.ylabel('借款笔数分布')
df_age['逾期占比'].plot(rot=45, color='steelblue', alpha=0.5, secondary_y=True)
plt.ylabel('逾期占比情况')
plt.grid(True)
plt.show()

# 学历与借款的情况
df_edu = pd.pivot_table(data=data, columns='标当前状态', index='学历认证', values='列表序号', aggfunc=np.size)
df_edu['借款笔数'] = df_edu.sum(axis=1)
df_edu['借款笔数占比'] = df_edu['借款笔数'] / df_edu['借款笔数'].sum()
df_edu['逾期占比'] = df_edu['逾期中'] / df_edu['借款笔数']

# 画图
plt.figure(figsize=(16, 9))
plt.subplot(121)
plt.pie(x=df_edu['借款笔数占比'], labels=['成功认证', '未成功认证'], colors=['orange', 'blue'], autopct='%.1f%%', pctdistance=0.5,
        labeldistance=1.1)
plt.title('学历认证比例')
plt.subplot(122)
plt.bar(x=df_edu.index, height=df_edu['逾期占比'], color=['orange', 'blue'], alpha=0.5)
plt.title('不同学历人群逾期情况')
plt.suptitle('不同学历人群客户画像')
plt.show()

# plotly.graph_objs交互式画图
# 画条形图
trace_basic = [go.Bar(x=df_edu.index, y=df_edu['逾期占比'], marker=dict(color='orange'), opacity=0.50)]
layout = go.Layout(title='不同学历人群逾期情况', xaxis=dict(title='不同学历人群客户画像'))
figure_basic = go.Figure(data=trace_basic, layout=layout, )
pyplot(figure_basic)

# 画饼图
trace_basic1 = [
    go.Pie(labels=['成功认证', '未成功认证'], values=df_edu['借款笔数占比'], hole=0.2, textfont=dict(size=12, color='white'))]
layout1 = go.Layout(title='学历认证比例')
figure_basic1 = go.Figure(data=trace_basic1, layout=layout1, )
pyplot(figure_basic1)


# 设计函数对多个对象进行处理
def trans(data, col, ind):
    df = pd.pivot_table(data=data, columns=col, index=ind, values='列表序号', aggfunc=np.size)
    df['借款笔数'] = df.sum(axis=1)
    df['借款笔数占比'] = df['借款笔数'] / df_edu['借款笔数'].sum()
    df['逾期占比'] = df['逾期中'] / df_edu['借款笔数']

    plt.figure(figsize=(16, 12))
    plt.subplot(121)
    plt.pie(x=df['借款笔数占比'], labels=['成功认证', '未成功认证'], colors=['orange', 'blue'], autopct='%.1f%%', pctdistance=0.5,
            labeldistance=1.1)
    plt.title('%s占比' % ind)
    plt.subplot(122)
    plt.bar(x=df.index, height=df['逾期占比'], color=['orange', 'blue'], alpha=0.5)
    plt.title('不同%s人群逾期情况' % ind)
    plt.suptitle('不同%s人群客户画像' % ind)
    plt.show()
    return df


trans(data, '标当前状态', '淘宝认证')
trans(data, '标当前状态', '征信认证')
trans(data, '标当前状态', '视频认证')
