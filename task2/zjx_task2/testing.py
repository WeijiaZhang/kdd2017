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

a = np.load('/home/wuxing/KDD/predict_task2.npz')
dta=a["arr_0"]
pdb.set_trace()
# tem = map(lambda x: np.array(x), a["arr_0"][:12])
# data = []
# for x in tem:
# 	data.append(x[-6:])
# data = np.array(data)

# print data
# data[:12][:] +=1
# print data

# new_data = old_data[:12, -6:]