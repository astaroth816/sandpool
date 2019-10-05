# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:50:17 2019

@author: andy3
"""


import matplotlib.pyplot as plt
import numpy as np
import math
t = np.linspace(0,math.pi, 1000)
x = np.sin(t)*np.sin(t)*np.sin(t)*16
y = np.cos(t)*13-np.cos(2*t)*5-np.cos(3*t)*2-np.cos(3.5*t)
plt.plot(x, y, color='red', linewidth=2)
plt.plot(-x, y, color='red', linewidth=2)
s = np.linspace(0,math.pi, 1000)
z = 1.5*(np.sin(t)*np.sin(t)*np.sin(t)*16)
w = 2+1.5*(np.cos(t)*13-np.cos(2*t)*5-np.cos(3*t)*2-np.cos(3.5*t))
plt.plot(z, w, color='red', linewidth=2)
plt.plot(-z, w, color='red', linewidth=2)
plt.ylim(-40, 50)
plt.xlim(-50, 50)
s = np.linspace(0,math.pi, 1000)
z = 2*(np.sin(t)*np.sin(t)*np.sin(t)*16)
w = 4+2*(np.cos(t)*13-np.cos(2*t)*5-np.cos(3*t)*2-np.cos(3.5*t))
plt.plot(z, w, color='red', linewidth=2)
plt.plot(-z, w, color='red', linewidth=2)
plt.ylim(-40, 50)
plt.xlim(-50, 50)
plt.show()