import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Bar, Funnel
from pyecharts import options as opts
import seaborn as sns
from ipykernel import kernelapp as app
############漏斗形流失分析

# # 解决suptitle报警问题
# import matplotlib
# matplotlib.use("TkAgg")

# 设置主题
plt.style.use('ggplot')
sns.set(style='darkgrid', font_scale=1.5)
# 解决中文字符显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_csv('tianchi_mobile_recommend_train_user.csv', dtype=str, encoding='utf-8')
data = data.sample(frac=0.2,replace=False) #抽样
# 统计缺失值
data.apply(lambda x: sum(x.isnull()))
# 统计缺失率
data.apply(lambda x: sum(x.isnull()) / len(x))
# 分割日期,转换形式
data['date'] = data['time'].str[:-3]
data['hour'] = data['time'].str[-2:].astype(int)
data['date'] = pd.to_datetime(data['date'])
data['time'] = pd.to_datetime(data['time'])
data.sort_values(by='time', ascending=True, inplace=True) #按时间排序
data.reset_index(drop=True, inplace=True)

# 对字符型数据进行统计，describe()理解include参数
data.describe(include=['object'])
pv_daily = data.groupby('date').count()[['user_id']].rename(columns={'user_id':'pv'})
pv_daily.head()
# 每日独立访客量
uv_daily = data.groupby('date')[['user_id']].apply(lambda x: x.drop_duplicates().count()).rename(columns={'user_id':'uv'})
uv_daily.head()
# 合并
pv_uv_daily = pd.concat([pv_daily, uv_daily], axis=1)
pv_uv_daily.head()
# pv与uv的相关性，method可以是相关性{spearman、pearson（默认）} 非相关性{kendall}
pv_uv_daily.corr(method='pearson')

plt.figure(figsize=(9, 9), dpi=70)
plt.subplot(211)
plt.plot(pv_daily, color='red')
plt.title('每日访问量', pad=10)
plt.xticks(rotation=45)
plt.grid(b=False)
plt.subplot(212)
plt.plot(uv_daily, color='green')
plt.title('每日访问用户数', pad=10)
plt.xticks(rotation=45)
plt.suptitle('PV和UV变化趋势', fontsize=20)
plt.subplots_adjust(hspace=0.5)
plt.grid(b=False)
plt.show()
#时PV和时UV
pv_hour = data.groupby('hour').count()[['user_id']].rename(columns={'user_id':'pv'})
uv_hour = data.groupby('hour')[['user_id']].apply(lambda x: x.drop_duplicates().count()).rename(columns={'user_id':'uv'})
pv_uv_hour = pd.concat([pv_hour, uv_hour], axis=1)
pv_uv_hour.head()

# 相关性
pv_uv_hour.corr(method='spearman')

fig = plt.figure(figsize=(9, 7), dpi=70)
fig.suptitle('PV和UV变化趋势', y=0.93, fontsize=18)
ax1 = fig.add_subplot(111)
ax1.plot(pv_hour, color='blue', label='每小时访问量')
ax1.set_xticks(list(np.arange(0,24)))
ax1.legend(loc='upper center', fontsize=12)
ax1.set_ylabel('访问量')
ax1.set_xlabel('小时')
ax1.grid(False)
ax2 = ax1.twinx()
ax2.plot(uv_hour, color='red', label='每小时访问用户数')
ax2.legend(loc='upper left', fontsize=12)
ax2.set_ylabel('访问用户数')
ax2.grid(False)
fig.show()
#不同行为类型用户PV分析
# data.groupby(['date', 'behavior_type'])[['user_id']].count().reset_index().rename(columns={'user_id':'pv'})
diff_behavior_pv = data.pivot_table(columns='behavior_type', index='date', values='user_id', aggfunc='count').rename(columns={'1':'click', '2':'collect', '3':'addToCart', '4':'pay'}).reset_index()
diff_behavior_pv.describe()
diff_behavior_pv.head()


bar_width=0.2
xticklabels = ['7-%d' % i for i in list(np.arange(18,31))] + ['8-%d' % i for i in list(np.arange(1, 24))]
plt.figure(figsize=(20, 9))
plt.bar(diff_behavior_pv.index-2*bar_width, diff_behavior_pv.click, width=bar_width, label='click')
plt.bar(diff_behavior_pv.index-bar_width, diff_behavior_pv.collect, bottom=0, width=bar_width, color='red', alpha=0.5, label='collect')
plt.bar(diff_behavior_pv.index, diff_behavior_pv.addToCart, bottom=0, width=bar_width, color='black', label='toCart')
plt.bar(diff_behavior_pv.index+bar_width, diff_behavior_pv.pay, bottom=0, width=bar_width, color='blue',  label='pay')
plt.yscale('log')
plt.yticks(fontsize=20)
plt.xticks(ticks=list(np.arange(0, 37, 3)), labels=xticklabels[::3], rotation=45, fontsize=20)
plt.xlabel('日期', fontsize=22)
plt.ylabel('浏览量', fontsize=22)
plt.title('每天不同行为类型用户PV情况', fontsize=36)
plt.legend(loc='best', fontsize=18)
plt.grid(False)
plt.savefig('每天不同行为类型用户PV情况.png', quality=95, dpi=70)
plt.show()

#操作行为情况
pv_detatil = data.pivot_table(columns='behavior_type', index='hour', values='user_id', aggfunc=np.size)
pv_detatil.rename(columns={'1':'click', '2':'collect', '3':'addToCart', '4':'pay'}, inplace=True)
pv_detatil.head()
#操作行为可视化
for i in pv_detatil.columns.tolist()[1:]:
    plt.plot(pv_detatil[i], label=i)
plt.legend(loc='best', fontsize=12)
plt.title('访问行为情况', fontsize=18)
plt.xticks(list(np.arange(0, 24)))
plt.xlabel('小时')
plt.ylabel('数量')
plt.grid()
plt.show()

data_user_buy = data[data.behavior_type == '4'].groupby('user_id').size()
data_user_buy.head()

click_times = data[data.behavior_type == '1'].groupby('user_id').size()
collect_times = data[data.behavior_type == '2'].groupby('user_id').size()
addToCart_times = data[data.behavior_type == '3'].groupby('user_id').size()
pay_times = data[data.behavior_type == '4'].groupby('user_id').size()
user_behavior = pd.concat([click_times, collect_times, addToCart_times, pay_times], axis=1)
user_behavior.columns= ['click', 'collect', 'addToCart','pay']
user_behavior.fillna(0, inplace=True)
user_behavior['pay_per_click'] = round(user_behavior['click'] / user_behavior['pay'], 1)
user_behavior.head()

#user_behavior.describe()

plt.hist(user_behavior[(user_behavior.pay_per_click < 800) & (user_behavior.pay_per_click >=0)].pay_per_click, bins=30)
plt.show()
# 相关性
user_behavior.corr(method='spearman').iloc[3:4, :3]

#用户消费行为分析
#日ARPU和日ARPPU
# 每日活跃用户数
active_user_daily = data.groupby('date')[['user_id']].apply(lambda x: x.drop_duplicates().count())
active_user_daily.head()

# 每日付费用户数
pay_user_daily = data[data.behavior_type == '4'].groupby('date')[['user_id']].apply(lambda x: x.drop_duplicates().count())
pay_user_daily.head()

# 合并
consume_daily = pd.concat([active_user_daily, pay_user_daily], axis=1)
# 重新命名字段
consume_daily.columns= ['activeUserDaily', 'payUserDaily']
# 由于数据中没有给用户消费金额，设每日每位用户消费为500
consume_daily['totalIncome'] = 500
# 计算ARPU
consume_daily['ARPU'] = round(consume_daily['totalIncome'] * consume_daily['payUserDaily'] / consume_daily['activeUserDaily'], 3)
# 计算ARPPU
consume_daily['ARPPU'] = round(consume_daily['totalIncome'] * consume_daily['payUserDaily'] / consume_daily['payUserDaily'])
consume_daily.head()

fig=plt.figure(figsize=(12, 8), dpi=100)
fig.suptitle('日用户消费行为', fontsize=20)
ax1 = fig.add_subplot(111)
ax1.plot(consume_daily['ARPU'],'ro-', label='日ARPU')
ax1.grid()
ax1.set_yticklabels(labels=list(np.arange(100, 300, 20)), fontsize=14)
ax1.set_ylabel('ARPU',fontsize=16)
ax1.legend(fontsize=14)
ax1.set_xlabel('日期', fontsize=16)
ax2 = ax1.twinx()
ax2.plot(consume_daily['ARPPU'], 'b-', label='日ARPPU')
ax2.legend(loc='upper left', fontsize=14)
ax2.set_yticklabels(labels=list(np.arange(470, 600, 10)), fontsize=14)
ax2.set_ylabel('ARPPU',fontsize=16)
fig.savefig('用户日消费行为.png', dpi=70, quality=95)
fig.show()



#用户购买次数情况分析
data['operation'] = 1
customer_operation = data.groupby(['date', 'user_id', 'behavior_type'])[['operation']].count()
customer_operation.reset_index(level=['date', 'user_id', 'behavior_type'], inplace=True)
customer_operation.head()
customer_operation[customer_operation.behavior_type == '4']['operation'].describe()
# 购买次数超过50次的用户数
# customer_operation[(customer_operation.behavior_type == '4') & (customer_operation.operation > 50)].count()['user_id']
# plt.hist(customer_operation[(customer_operation.behavior_type == '4') & (customer_operation.operation < 50)].operation, bins=10)
# plt.show()

#每天平均消费次数
customer_operation.groupby('date').apply(lambda x: x[x.behavior_type == '4'].operation.sum() / len(x.user_id.unique())).plot()
#活跃用户付费率
customer_operation.groupby('date').apply(lambda x: x[x.behavior_type == '4'].operation.count() / len(x.user_id.unique())).plot()
#同一时间段用户消费次数分布
customer_hour_operation = data[data.behavior_type == '4'].groupby(['user_id', 'date', 'hour'])[['operation']].sum()
customer_hour_operation.reset_index(level=['user_id', 'date', 'hour'], inplace=True)
customer_hour_operation.head()
customer_hour_operation.operation.max()
# plt.scatter(customer_hour_operation.hour, customer_hour_operation.operation)
# plt.xlabel('hour', fontsize=14)
# plt.ylabel('buy times', fontsize=14)
# plt.show()


#复购行为分析
#月复购率
# 按周期
data_rebuy = data[data.behavior_type == '4'].groupby('user_id')['date'].apply(lambda x: len(x.unique()))
data_rebuy[:5]
# 复购率
data_rebuy[data_rebuy >= 2].count() / data_rebuy.count()
data_day_buy = data[data.behavior_type == '4'].groupby(['user_id']).date.apply(lambda x: x.sort_values()).diff(1).dropna().map(lambda x: x.days)


#留存率
from datetime import datetime
day_user = {}
for dt in set(data.date.dt.strftime('%Y%m%d').values.tolist()):
    user = list(set(data[data.date == datetime(int(dt[:4]),int(dt[4:6]),int(dt[6:]))]['user_id'].values.tolist()))
    day_user.update({dt:user})
# 由于字典是无序的，需按日期排序
day_user = sorted(day_user.items(), key=lambda x:x[0], reverse=False)
# 计算每日新增用户
a = {}
t = set(day_user[0][1])
a.update({'20141118':t})
for i in day_user[1:]:
    j = (set(i[1]) - t)
    a.update({i[0]:j})
    t = t | set(i[1])
# 目的是为了和day_user类型一样
a = sorted(a.items(), key=lambda x:x[0], reverse=False)
# 计算留存
retention = {}
ls = []
for i, k in enumerate(a):
    ls.append(len(k[1]))
    for j in day_user[i+1:]:
        li = len(set(k[1]) & set(j[1]))
        ls.append(li)
    retention.update({k[0]: ls})
    ls = []

# 目的是为了和day_user类型一样
retention = sorted(retention.items(), key=lambda x:x[0], reverse=False)
re = {}
for i in retention[:16]:
    re.update({i[0]: i[1][:15]})
retention = pd.DataFrame(re)

retention = retention.T
retention.drop([8,9,11,12,13], axis=1, inplace=True)
retention.columns = ['新增用户', '次日留存', '2日留存','3日留存', '4日留存','5日留存', '6日留存', '7日留存', '10日留存','14日留存']

div = retention.columns.tolist()[:-1]
for i, dived in enumerate(retention.columns.tolist()[1:]):
    retention['{}率'.format(dived)] = round(retention[dived] / retention['新增用户'], 3)
cols=['新增用户','次日留存','次日留存率','2日留存','2日留存率','3日留存','3日留存率','4日留存','4日留存率','5日留存','5日留存率','6日留存','6日留存率','7日留存','7日留存率','10日留存','10日留存率','14日留存','14日留存率']
retention = retention[cols]
retention.sort_index(inplace=True)
retention.head()


retention.index.str[-4:]
plt.figure(figsize=(16, 9), dpi=90)
x = [i[:2] + '-' + i[2:] for i in retention.index.str[-4:].tolist()]
y1 = retention['次日留存率']
y2 = retention['3日留存率']
y3 = retention['7日留存率']
y4 = retention['10日留存率']
y5 = retention['14日留存率']

plt.plot(x, y1, 'ro-', label='次日留存率')
plt.plot(x, y2, 'bo-', label='3日留存率')
plt.plot(x, y3, 'yo--', label='5日留存率')
plt.plot(x, y4, 'gd-', label='7日留存率')
plt.plot(x, y1, 'rd-', label='10日留存率')
plt.plot(x, y5, 'cd--', label='14日留存率')

plt.legend(loc='best')
plt.title('14天内用户留存率情况', fontsize=30)
plt.xlabel('日期', fontsize=20)
plt.ylabel('留存率', fontsize=20)
plt.show()


data_user_count = data.groupby('behavior_type').size()
data_user_count
pv_all = data.user_id.count()
pv_all
pv_click = (pv_all - data_user_count[0]) / pv_all
click_cart = 1 - (data_user_count[0] - data_user_count[2]) / data_user_count[0]
cart_collect = 1 - (data_user_count[2] - data_user_count[1]) / data_user_count[2]
collect_pay = 1 - (data_user_count[1] - data_user_count[3]) / data_user_count[1]
cart_pay = 1 - (data_user_count[2] - data_user_count[3]) / data_user_count[2]
change_rate = pd.DataFrame({'计数': [pv_all, data_user_count[0], data_user_count[2], data_user_count[3]],\
                            '单一转化率':[1, pv_click, click_cart, cart_pay]}, index=['浏览', '点击', '加入购物车', '支付'])
change_rate['总体转化率'] = change_rate['计数'] / pv_all
change_rate



#二八理论分析淘宝商品
goods_category = data[data.behavior_type == '4'].groupby('item_category')[['user_id']].count().rename(columns={'user_id':'购买量'}).sort_values(by='购买量', ascending=False)
goods_category['累计购买量'] = goods_category.cumsum()
goods_category['占比'] = goods_category['累计购买量'] / goods_category['购买量'].sum()
goods_category['分类'] = np.where(goods_category['占比'] <= 0.80, '产值前80%', '产值后20%')
goods_pareto = goods_category.groupby('分类')[['购买量']].count().rename(columns={'购买量':'商品数'})
goods_pareto['商品数占比'] = round(goods_pareto['商品数'] / goods_pareto['商品数'].sum(), 3)
goods_pareto

#用户细分（RFM）
#计算R
from datetime import datetime
recent_user_buy = data[data.behavior_type == '4'].groupby('user_id')['date'].apply(lambda x: datetime(2014, 12, 20)-x.sort_values().iloc[-1])
recent_user_buy = recent_user_buy.reset_index()
recent_user_buy.columns = ['user_id', 'recent']
recent_user_buy.recent = recent_user_buy.recent.map(lambda x: x.days)
recent_user_buy.head()
#计算F
buy_freq = data[data.behavior_type == '4'].groupby('user_id').date.count()
buy_freq = buy_freq.reset_index().rename(columns={'date': 'freq'})
buy_freq.head()
rfm = pd.merge(recent_user_buy, buy_freq, right_on='user_id', left_on='user_id')
rfm.head()
#给予指标
rfm['R_value'] = pd.qcut(rfm.recent, 2, labels=['高', '低'])
rfm['F_value'] = pd.qcut(rfm.freq, 2, labels=['低', '高'])
rfm['rf'] = rfm['R_value'].str.cat(rfm.F_value)
rfm.head()
#用户分类
def trans_value(x):
    if x == '高高': return '价值用户'
    elif x == '高低': return '发展用户'
    elif x == '低高': return '挽留客户'
    else: return '潜在客户'
rfm['rank'] = rfm.rf.apply(trans_value)
rfm.head()

#统计不同类型用户结果及可视化
rfm.groupby('rank')[['user_id']].count()
plt.pie(rfm.groupby('rank')[['user_id']].count().values.tolist(), labels=rfm.groupby('rank')[['user_id']].count().index.tolist(), shadow=True, autopct='%.1f%%', radius=1.5, textprops=dict(fontsize=12))
plt.title('用户分类情况', fontsize=30, pad=45, color='blue')
plt.show()

# pd.set_option('display.max_columns', None)#显示全部结果
# pd.set_option('display.max_rows', None)







