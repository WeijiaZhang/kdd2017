import numpy as np
from scipy import interpolate
import pylab as pl

x = np.linspace(0,10,11)
y = np.sin(x)

xnew = np.linspace(0,10,101)
print xnew
