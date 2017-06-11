import numpy as np
import os,sys
import scipy.io as scio
import pandas as pd
import pyflux as pf
from scipy import stats
from scipy import interpolate
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
from datetime import datetime,timedelta,time,date




data=pd.read_csv('/home/wuxing/KDD/task2_test.csv')
data.id=data['tollgate_id'].values
data.day=data['time']
data.direction=data['direction'].values
data.vol=data['volume'].values
num=len(data)


for i in range(len(data)):
	data.day[i]=data.day[i].replace('06:','08:')
	data.day[i]=data.day[i].replace('07:','09:')
	data.day[i]=data.day[i].replace('15:','17:')
	data.day[i]=data.day[i].replace('16:','18:')


pdb.set_trace()
#--------------------------------------parallel--------------------------#
dat=[]
a = np.load('/home/wuxing/KDD/predict_one.npz')
tem1 = map(lambda x: np.array(x), a["arr_0"][:13])
data1 = []
for x in tem1:
	data1.append(x[-6:])
	mid1 = map(lambda y: int(round(y)), x[-6:])
	dat.append(mid1)
	mid1=None
data1 = np.array(data1)
pdb.set_trace()

x = None 
b = np.load('/home/wuxing/KDD/predict_only13.npz')
tem2 = map(lambda x: np.array(x), b["arr_0"][13:14])
data2 = []
for x in tem2:
	data2.append(x[-6:])
	mid2 = map(lambda y: int(round(y)), x[-6:])
	dat.append(mid2)
	mid2=None
data2 = np.array(data2)
pdb.set_trace()

x = None 
c = np.load('/home/wuxing/KDD/predict_f14.npz')
tem3 = map(lambda x: np.array(x), c["arr_0"][14:58])
data3 = []
for x in tem3:
	data3.append(x[-6:])
	mid3 = map(lambda y: int(round(y)), x[-6:])
	dat.append(mid3)
	mid3=None
data3 = np.array(data3)
pdb.set_trace()


x = None 
d = np.load('/home/wuxing/KDD/predict_f58.npz')
tem4 = map(lambda x: np.array(x), d["arr_0"][58:70])
data4 = []
for x in tem4:
	data4.append(x[-6:])
	mid4 = map(lambda y: int(round(y)), x[-6:])
	dat.append(mid4)
	mid4 = None
data4 = np.array(data4)
pdb.set_trace()


#--------------------------------------parallel--------------------------#

Data=[]
for i in range(len(dat)):
	for j in range(len(dat[0])):
		Data.append(dat[i][j])

Data = np.array(Data)
print 'Done'
data.vol = Data
np.savez('/home/wuxing/KDD/predict_task2.npz',data)
pdb.set_trace()