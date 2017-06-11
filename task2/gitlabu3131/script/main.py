# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:42:21 2017
@author: LiuYangkai
"""

from datarepo import Repo
from time import clock
import dataproc, features, xgboost, logging
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
def main(params={}, debug=False):
    r = Repo()
    df1 = r(features.trivial_feature, name='train')
    df1['is_work'] = df1.time.apply(features.is_work_day)
    df2 = r(features.last_2hour, name='train')
    df = r(lambda x,y:x.merge(y, 
            on=['tollgate_id', 'direction', 'time'],
            how='left').reindex_axis(['tollgate_id', 'direction',
            'time', 'hour', 'max_vol', 'min_vol', 'avg_vol', 'last2h', 
            'is_work', 'volume'], axis='columns'), 'train', df1, df2)
    test_index, train_index = dataproc.select_test(
                    df.shape[0], round(df.shape[0]*0.1))
    train_feature = df.ix[train_index, 'hour':'is_work'].reset_index(drop=True)
    train_label = df.ix[train_index, 'volume'].reset_index(drop=True)
    model = xgboost.XGBRegressor(**params)
    #model = KNeighborsRegressor()
    logging.info('开始交叉验证...')
    scores = cross_val_score(model, train_feature, train_label, 
                                     cv=KFold(n_splits=3, shuffle=False), 
                                     #n_jobs=-1, 
                                     scoring=dataproc.official_loss
                                     )
    logging.info('交叉验证结果: %s.'%scores)
    if debug:
        return
    logging.info('用所有训练数据训练模型...')
    model.fit(train_feature, train_label)
    logging.info('模型训练完毕.')
    logging.info('开始测试...')
    test_feature = df.ix[test_index, 'hour':'is_work'].reset_index(drop=True)
    test_label = df.ix[test_index, 'volume'].reset_index(drop=True)
    y = model.predict(test_feature)
    logging.info('测试结果: %f.', dataproc.MAPE(y, test_label))
    res_feature = dataproc.get_predict_feature()
    
    res_y = model.predict(res_feature.ix[:, 'hour':'is_work'])
    res = res_feature.ix[:,'tollgate_id':'time']
    res['volume'] = res_y
    #dataproc.formatResult(res, name='xgboost_workingday')
if __name__ == '__main__':
    logging.basicConfig(
            level    = logging.DEBUG,
            format   = '%(asctime)s %(filename)s[line:%(lineno)d] \
                        %(levelname)s %(message)s',
            datefmt  = '%y%m%d %H:%M:%S',
            filename = '../temp/log.txt',
            filemode = 'w');
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    console.setFormatter(logging.Formatter('%(asctime)s %(filename)s: \
                                            %(levelname)-8s %(message)s'));
    logging.getLogger('').addHandler(console);
    clock()
    main()
    logging.info('共耗时%f分钟.' % (clock()/60))