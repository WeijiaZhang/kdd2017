# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:43:33 2017

@author: LiuYangkai
"""
import pandas as pd
import numpy as np
from datarepo import Repo
from math import floor
from datetime import datetime, timedelta
import dataproc
import logging


def volume_grouped(phase='train'):
    def round_time(t):
        m = int(floor(int(t[14:16]) / 20) * 20)
        return '%s:%s:00' % (t[:13], str(m) if m > 9 else '0%d' % m)
    r = Repo()
    path = '../../../dataset/training/volume(table 6)_training.csv'
    dt_start = '2016-09-19'
    dt_end = '2016-10-17'
    if phase != 'trian':
        path = '../../../dataset/testing_phase1/volume(table 6)_test1.csv'
        dt_start = '2016-10-18'
        dt_end = '2016-10-24'
    df = r(dataproc.prep_volume, name=phase, path=path,
           dt_start=dt_start, dt_end=dt_end)
    df = df.ix[:, [0, 1, 2]]
    df['time'] = df.time.apply(round_time)
    df = df.groupby(['tollgate_id', 'direction', 'time']).size()
    df = df.reset_index()
    df = df.rename_axis({0: 'volume'}, axis='columns')
    # to-do 平滑节假日异常值
    return df


def trivial_feature():
    '''之前对应时间段的最大、最小和平均流量'''
    r = Repo()
    df = r(volume_grouped, name='train')
    df = df.sort_values('time', ascending=True)
    res = pd.DataFrame(columns=['tollgate_id', 'direction', 'time', 'hour',
                                'max_vol', 'min_vol', 'avg_vol', 'volume'])
    enum_cap = [(int(1), True), (int(1), False), (int(2), False),
                (int(3), True), (int(3), False)]
    eid = 0
    for tid, dirc in enum_cap:
        tdf = df[(df.tollgate_id == tid) &
                 (df.direction == dirc)]
        tdf = tdf.drop(['tollgate_id', 'direction'], axis='columns')
        tdf['date'], tdf['time'] = tdf.time.str[:10], tdf.time.str[11:]
        tdf_grp = tdf.groupby('time')
        eid += 1
        gid = 0
        for tm, sdf in tdf_grp:
            sdf = sdf.reset_index()
            gid += 1
            if sdf.shape[0] < 2:
                continue
            st_date = np.datetime64(sdf.ix[0, 'date'])
            sm = sdf.ix[0, 'volume']
            min_vol = sm
            max_vol = sm
            # logging.info('Lane:%d/%d. Group:%d/%d.' % (eid, len(enum_cap),
            # gid, len(tdf_grp.groups)))
            for k in range(1, sdf.shape[0]):
                nw_date = sdf.ix[k, 'date']
                span = int((np.datetime64(nw_date) - st_date) /
                           np.timedelta64(1, 'D'))
                if span <= 0:
                    logging.error('时间未排序!')
                    break
                avg_vol = sm / span
                t = sdf.ix[k, 'volume']
                dat = {'tollgate_id': tid, 'direction': dirc, 'time': '%s %s' % (nw_date, tm),
                       'hour': round((int(tm[:2]) * 60 + int(tm[3:5])) / 60) % 24,
                       'max_vol': max_vol, 'min_vol': min_vol, 'avg_vol': avg_vol,
                       'volume': t}
                res = res.append(dat, ignore_index=True)
                sm += t
                min_vol = min(min_vol, t)
                max_vol = max(max_vol, t)
    return res


def last_2hour(phase='train'):
    '''前两个小时的流量'''
    r = Repo()
    df = r(volume_grouped, name=phase, phase=phase)
    if phase == 'train':
        df_dp = df.drop('volume', axis='columns')
    df['time'] = df.time.apply(lambda t: pd.to_datetime(str(np.datetime64(t) +
                                                            np.timedelta64(2, 'h'))).strftime('%Y-%m-%d %H:%M:%S'))
    if phase == 'train':
        df = df_dp.merge(
            df, on=['tollgate_id', 'direction', 'time'], how='left')
        df = df.fillna(0)
    df = df.rename_axis({'volume': 'last2h'}, axis='columns')
    return df


def is_work_day(itm):
    '''判断所给的日期是否是工作日，格式是yyyy-mm-dd HH:MM:SS;；返回True或False。
        由apply函数调用'''
    d = itm[:10]
    if '2016-09-15' <= d and\
        d <= '2016-09-17' or\
        '2016-10-01' <= d and\
            d <= '2016-10-07':
        return False
    if d == '2016-09-18' or\
       d == '2016-10-08' or\
       d == '2016-10-09':
        return True
    w = datetime.strptime(d, '%Y-%m-%d').weekday()
    return 1 <= w and w <= 5
