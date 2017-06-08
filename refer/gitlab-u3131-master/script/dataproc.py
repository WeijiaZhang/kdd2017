# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 08:44:11 2017

@author: LiuYangkai
"""
import pandas as pd
import numpy as np
import logging, re, features
from datarepo import Repo
from random import randint
import matplotlib.pyplot as plt
def is_like(df, pat):
    f = lambda e,pat:re.match(pat, e) is not None
    return df.apply(f, pat=pat)
def is_in(df, vals):
    return df.isin(vals)
def is_null(df):
    return df.isnull()
def is_outlier(df):
    return np.abs(df - df.mean()) > \
        3*df.std()
def prep_volume(path, dt_start, dt_end):
    '''只是做了个检查，每个域是否是有效的，都通过了，就没再管了；
        更常规的思路是下面的prep_weather函数'''
    df = pd.read_csv(path)
    logging.info('%s 中原有数据%d条.'%(path, df.shape[0]))
    df = df[(df.time >= '%s 00:00:00'%dt_start) &\
        (df.time <= '%s 23:59:59'%dt_end)]
    df = df[(df.tollgate_id >= 1) &\
        (df.tollgate_id <= 3)]
    df = df[(df.direction >= 0) &\
        (df.direction <= 1)]
    df = df[(df.vehicle_model >= 0) &\
        (df.vehicle_model <= 7)]
    df = df[(df.has_etc >= 0) &\
        (df.has_etc <= 1)]
    df['vehicle_type'] = df.vehicle_type.fillna(2)
    df = df[(df.vehicle_type >= 0) &(df.vehicle_type <= 2)]
    #df['time'] = df.time.apply(lambda x:np.datetime64(x))    
    df['direction'] = df.direction.apply(lambda x:True if x == 1 else False)    
    df['has_etc'] = df.has_etc.apply(lambda x: True if x == 1 else False)
    logging.info('%s 处理后还剩%d条.'%(path, df.shape[0]))
    return df
def prep_weather(path):
    '''找出异常值所在的位置，然后人工修改，爆出来的只是可能是异常值，但具体的还得人工确定'''
    df = pd.read_csv(path)
    index = ~is_like(df.date, r'2016\-\d\d\-\d\d')
    if index.any():
        logging.info('date域格式不满足:%s'%(index[index].index.values+2))
    index = ~is_in(df.hour, [0,3,6,9,12,15,18,21])
    if index.any():
        logging.info('hour域不在范围內:%s'%(index[index].index.values+2))
    for k in ['pressure', 'sea_pressure', 'wind_direction',
              'wind_speed', 'temperature', 'rel_humidity',
              'precipitation']:
        index = is_null(df[k])
        if index.any():
            logging.info('%s域有null值:%s'%(k, (index[index].index.values+2)))
        index = is_outlier(df[k])
        if index.any():
            logging.info('%s域有异常值:%s'%(k, (index[index].index.values+2)))
    df['date'] = df.date.apply(lambda x:np.datetime64(x))
    return df
def select_test(n, count):
    '''从大小为n的样本中随机选择count个作为测试，其余的用来训练；
        返回测试索引和训练索引'''
    index = pd.Series([False for _ in range(n)])
    p = 0
    while p < count:
        ii = randint(0, n-1)
        if index[ii]:
            continue
        index[ii] = True
        p += 1
    return (index, ~index)
def MAPE(pred, truth):
    '''官方给的loss函数，因为自己模型的特点，把公式里的T当作1来处理'''
    if not isinstance(pred, pd.Series):
        pred = pd.Series(pred)
    if not isinstance(truth, pd.Series):
        truth = pd.Series(truth)
    pred = pred.reset_index(drop=True)
    truth = truth.reset_index(drop=True)    
    d = (pred - truth).abs()
    return (d / truth).mean()
def official_loss(estimator, X, y):
    yp = estimator.predict(X)
    return MAPE(yp, y)
def get_predict_feature():
    r = Repo()
    lst2 = r(features.last_2hour, name='pred', phase='pred')
    tvl = r(features.trivial_feature_v2, name='train')
    enum_cap = [(int(1), True), (int(1), False), (int(2), False), 
                (int(3), True), (int(3), False)]
    time_int = ['08:00:00', '08:20:00', '08:40:00', '09:00:00', '09:20:00',
                '09:40:00', '17:00:00', '17:20:00', '17:40:00', '18:00:00',
                '18:20:00', '18:40:00']
    dates = ['2016-10-%d'%k for k in range(18, 25)]
    res = pd.DataFrame(columns=['tollgate_id','direction','time','max_vol',
                                 'min_vol','avg_vol'])
    for tid,drc in enum_cap:
        for t in time_int:
            for d in dates:
                index = (tvl.tollgate_id == tid) & (tvl.direction == drc) &\
                        (tvl.time == '2016-10-17 %s'%t)
                max_vol = tvl[index].max_vol.values[0]
                min_vol = tvl[index].min_vol.values[0]
                avg_vol = tvl[index].avg_vol.values[0]
                data = {'tollgate_id':tid,'direction':drc,'time':'%s %s'%(d,t),
                        'max_vol':max_vol,'min_vol':min_vol,'avg_vol':avg_vol}
                res = res.append(data, ignore_index=True)
    f = res.merge(lst2, on=['tollgate_id','direction','time'], how='left')
    f['hour'] = f.time.apply(lambda t:round((int(t[11:13])*60+int(t[14:16]))/60)%24)
    f['is_work'] = f.time.apply(features.is_work_day)
    f = f.reindex_axis(['tollgate_id','direction','time','hour',
                        'max_vol','min_vol','avg_vol','last2h',
                        'is_work'],axis='columns')
    return f
def formatResult(df, name='basic'):
    def time_win(dt):
        day = dt[:10]        
        h = int(dt[11:13])
        m = int(dt[14:16])
        m += 20
        if m >= 60:
            h += 1
            m = 0
        return '[%s,%s %s)'%(dt,day,'%s:%s:00'%(str(h) \
                                if h >= 10 else '0%d'%h,
                                str(m) if m >= 10 else '0%d'%m))
    df['time'] = df.time.apply(time_win)
    df = df.rename_axis({'time':'time_window'}, axis='columns')
    df = df.reindex_axis(['tollgate_id','time_window','direction',
                          'volume'], axis='columns')
    df['direction'] = df.direction.astype(np.int)
    df['tollgate_id'] = df.tollgate_id.astype(np.int)
    df = df.sort_values(['tollgate_id','direction','time_window'])
    Repo().saveResult(df, name)
def hist_error(true, pred):
    if not isinstance(true, np.ndarray):
        true = true.values
    if not isinstance(pred, np.ndarray):
        pred = pred.values
    plt.figure()
    d = np.power((true - pred), 2).round()
    logging.info('MSE: %f~%f'%(np.min(d), np.max(d)))
    logging.info('argmax: %d'%np.argmax(d))
    plt.hist(d, bins=50)
    plt.show()