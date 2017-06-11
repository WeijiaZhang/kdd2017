import numpy as np
import os,sys
import scipy.io as scio
import pandas as pd
import pyflux as pf
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import pdb
import test_stationarity
from statsmodels.tsa.arima_model import ARMA
from dateutil.relativedelta import relativedelta
from copy import deepcopy

class arima_model:

    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxint

    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    def certain_model(self, p, q):
            model = ARMA(self.data_ts, order=(p, q))
            try:
                self.properModel = model.fit( disp=-1, method='css')
                self.p = p
                self.q = q
                self.bic = self.properModel.bic
                self.predict_ts = self.properModel.predict()
                self.resid_ts = deepcopy(self.properModel.resid)
            except:
                print 'You can not fit the model with this parameter p,q'

    def forecast_next_day_value(self, type='day'):
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params

        # print self.properModel.params
        if self.p == 0:   # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)

        predict_value = np.dot(para[1:], values) + self.properModel.constant[0]
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat

    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)

rcParams['figure.figsize'] = 15,6


class item:
	def __init__(self):
		self.id=[]
		self.dir=[]
		self.day=[]
		self.list=[]

subdata=item()


data=pd.read_csv('/home/wuxing/KDD/task2_test.csv')
data.id=data['tollgate_id'].values
data.day=data['time']
data.direction=data['direction'].values
data.vol=data['volume'].values
num=len(data)


for i in range(num//6):
	subdata.id.append(data.id[i*6])
	subdata.dir.append(data.direction[i*6])
	subdata.day.append((data.day[i*6][8]+data.day[i*6][9]+
		data.day[i*6][11]+data.day[i*6][12]))
	vol=[data.vol[i*6],data.vol[i*6+1],data.vol[i*6+2],
	data.vol[i*6+3],data.vol[i*6+4],data.vol[i*6+5]]
	subdata.list.append(vol)
	vol=[];


# dta = pd.Series(subdata.list[0])
dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]
dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1','90'))
dta.plot(figsize=(12,8))
plt.show()



# test_stationarity.testStationarity(dta)
test_stationarity.draw_acf_pacf(dta,l=10) 
dta_log = np.log(dta)
test_stationarity.draw_ts(dta_log)
test_stationarity.draw_trend(dta_log,12)


def diff_ts(ts, d):
    global shift_ts_list
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print last_data_shift_list
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts


def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data


diffed_ts = diff_ts(dta_log,d=[12,1])
pdb.set_trace()
test_stationarity.testStationarity(diffed_ts)
test_stationarity.draw_acf_pacf(diffed_ts,l=3) 
model = arima_model(diffed_ts)
pdb.set_trace()
model.get_proper_model()
print 'bic:',model.bic,'p:',model.p,'q:',model.q
# print model.properModel.forecast()[0]
# print model.forecast_next_day_value(type='day')

# model2=ARMA(diffed_ts,(1,1,1)).fit()
model2=ARMA(dta_log,(2,1,1)).fit()
model2.summary2()
predict_sunspots = model2.predict('6','7',dynamic=True)
a = model2.forecast(5)[0]
a_ts = predict_diff_recover(a,d=[1,1])
log_a = np.exp(a_ts)



# pdb.set_trace()
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(111)
# diff1 = dta.diff(1)
# diff1.plot(ax=ax1)
# plt.show()

# dta = dta.diff(3)
# fig = plt.figure(figsize=(12,8))
# # xmajorLocator = MultipleLocator(1.0)
# ax1 = fig.add_subplot(211)
# # ax1.xaxis.set_major_locator(xmajorLocator)
# fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# # ax2.xaxis.set_major_locator(xmajorLocator)
# fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
# plt.show()
# pdb.set_trace()



# # print sm.stats.durbin_watson(arma_mod20.resid.values)
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(),lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid.values.squeeze(),lags=40,ax=ax2)
# plt.show()

# resid = arma_mod20.resid
# fig = plt.figure(figsize = (12,8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid,line='q',ax=ax,fit=True)




# model=pf.ARIMA(data=subdata,ar=5,ma=5,target='sunspot.year',family=pf.Normal())

# plt.figure(figsize=(15,5))
# plt.plot(data.index,data['sunspot.year'])
# plt.ylabel('Sunspots')
# plt.title('Year');
# plt.show()


# model=pf.ARIMA(data=data,ar=5,ma=5,target='sunspot.year',family=pf.Normal())

# x=model.fit("MLE")
# pdb.set_trace()
# x.summary()

# model.plot_z(figsize=(15,5))
# model.plot_fit(figsize=(15,10))

# # model.plot_predict_is(h=50,figsize=(15,5))
# model.plot_predict(h=5,past_values=20,figsize=(5,5))

# result = model.predict(h=5)


# data1=data['sunspot.year']
# import statsmodels.api as sm
# arma=sm.tsa.ARMA(data,order=(4,4));
# result=arma.fit(full_output=False,disp=0);

# pred=results.predict();
