# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:47:29 2017

@author: LiuYangkai
"""
import pandas as pd
from math import floor


def chg_time(tm):
    t = tm[11:]
    if t < '08:00:00' or\
        (t >= '10:00:00' and
         t < '17:00:00') or\
            t >= '19:00:00':
        return '00:00:00'
    m = int(t[3:5])
    m = int(floor(m / 20) * 20)
    return '%s:%s:00' % (t[:2], str(m) if m >= 10 else '0%d' % m)


def time_window(tm, day='2016-10-18'):
    h = int(tm[:2])
    m = int(tm[3:5])
    m += 20
    if m >= 60:
        h += 1
        m = 0
    return '[%s %s,%s %s)' % (day, tm, day, '%s:%s:00' % (str(h) if h >= 10 else '0%d' % h,
                                                          str(m) if m >= 10 else '0%d' % m))


def average():
    train_path = '../dataset/training/trajectories(table 5)_training.csv'
    dat = pd.read_csv(train_path)
    dat.ix[:, 'starting_time'] = dat.starting_time.apply(chg_time)
    dat = dat[dat.starting_time != '00:00:00']
    if {'vehicle_id', 'travel_seq'}.issubset(dat.columns):
        dat = dat.drop(['vehicle_id', 'travel_seq'], axis=1)
    avg = dat.groupby(
        ['intersection_id', 'tollgate_id', 'starting_time']).mean()
    avg = avg.reset_index()
    avg = avg.rename_axis({'travel_time': 'avg_travel_time'}, axis='columns')
    ret = pd.DataFrame()
    for k in ['2016-10-%d' % e for e in range(18, 25)]:
        tmp = avg.copy()
        tmp['starting_time'] = tmp.starting_time.apply(time_window, day=k)
        ret = ret.append(tmp)
    ret = ret.rename_axis({'starting_time': 'time_window'}, axis='columns')
    return ret.reindex_axis(['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'],
                            axis='columns')


def main():
    test_pred = average()
    test_pred.to_csv('baseline1.csv', index=False)
if __name__ == '__main__':
    main()
