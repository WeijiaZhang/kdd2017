# -*- coding:utf-8 -*-
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ÒÆ¶¯ÆœŸùÍŒ
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # ¶ÔsizežöÊýŸÝœøÐÐÒÆ¶¯ÆœŸù
    rol_mean = timeSeries.rolling(window=size).mean()
    # ¶ÔsizežöÊýŸÝœøÐÐŒÓÈšÒÆ¶¯ÆœŸù
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
¡¡¡¡Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def testStationarity(ts):
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    return dfoutput

# ×ÔÏà¹ØºÍÆ«Ïà¹ØÍŒ£¬Ä¬ÈÏœ×ÊýÎª31œ×
def draw_acf_pacf(ts, l):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=l, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=l, ax=ax2)
    plt.show()