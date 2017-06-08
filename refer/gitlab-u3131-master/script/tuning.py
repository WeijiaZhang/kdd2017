# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:42:21 2017

@author: LiuYangkai
"""
import main
def adjust():   
    params = {'max_depth':10, 'subsample':1.0, 'min_child_weight':1.0,
              'colsample_bytree':1.0, 'learning_rate':0.1, 'silent':False}
    #params = {'n_neighbors':5, 'weights':'uniform'}
    count = 0
    while len(input('输入任意字符停止调试:')) == 0:
        print('%d%s'%(count, '>'*10))
        for key, val in params.items():
            v = input('%s=%s, 改为(Enter不变):'%(key, val))
            if len(v) > 0:
                params[key] = type(val)(v)
        main.main(params, debug=True)
        print('<'*10)
        count += 1
if __name__ == '__main__':
    adjust()
