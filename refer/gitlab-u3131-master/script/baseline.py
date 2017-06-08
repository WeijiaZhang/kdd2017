# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:47:29 2017

@author: LiuYangkai
"""
import pandas as pd
from math import floor
from datarepo import Repo


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
    dat = pd.read_csv('../input/dataSets/training/volume(table 6)_training.csv',
                      usecols=[0, 1, 2])
    dat.ix[:, 'time'] = dat.time.apply(chg_time)
    dat = dat[dat.time != '00:00:00']
    avg = dat.groupby(['tollgate_id', 'direction', 'time']).size()
    avg = avg / 39
    avg = avg.reset_index()
    avg = avg.rename_axis({0: 'volume'}, axis='columns')
    ret = pd.DataFrame()
    for k in ['2016-10-%d' % e for e in range(18, 25)]:
        tmp = avg.copy()
        tmp.ix[:, 'time'] = tmp.time.apply(time_window, day=k)
        ret = ret.append(tmp)
    ret = ret.rename_axis({'time': 'time_window'}, axis='columns')
    return ret.reindex_axis(['tollgate_id', 'time_window', 'direction', 'volume'],
                            axis='columns')


def main():
    r = Repo()
    r.saveResult(r(average), name='baseline')
if __name__ == '__main__':
    average()
