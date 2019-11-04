# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:14:13 2019

@author: andy3
"""

import matplotlib.pyplot as plt 
import numpy as np

fig = plt.figure(edgecolor=(0,0,0),linewidth=10)
fig.suptitle('Love Polynomial,haha', fontsize=14, fontweight='bold')

#i
y = np.linspace(0, 10, 10)
# x are zeros
x = np.zeros(len(y))
ax = fig.add_subplot(2,8,1)  # add I graph
ax.plot(x, y, 'r')

#l
x = np.linspace(0.1, 5, 100)
# compute y = 1/10X^2
y = 1/(10 * x**2)

ax = fig.add_subplot(2,8,3) # add L graph
ax.get_yaxis().set_visible(False) # disable y axis
ax.plot(x, y, 'r')
##################

#o
theta = np.linspace(0, 2*np.pi, 100)
# the radius of the circle
r = 3
# compute x = 3cos(theta)
# compute y = 3sin(theta)
x = r*np.cos(theta)
y = r*np.sin(theta)

ax = fig.add_subplot(2,8,4) # add O graph
ax.get_yaxis().set_visible(False)	# disable y axis
ax.plot(x, y, 'r')
##################

#v
x = np.linspace(-1.5, 1.5, 100)
# compute y = |-3x|
y =np.abs(-3*x)

ax = fig.add_subplot(2,8,5) # add V graph
ax.get_yaxis().set_visible(False) # disable y axis
ax.plot(x, y, 'r')
##################

#E
y = np.linspace(0, 6, 100)
# compute x = -3|sin(y)|
x = -3*np.abs(np.sin(y))

ax = fig.add_subplot(2,8,6) # add E graph
ax.get_yaxis().set_visible(False) # disable y axis
ax.plot(x, y, 'r')
##################

#U
x = np.linspace(-3, 3, 100)
# compute x
y = x**4

ax = fig.add_subplot(2,8,8) # add U graph
ax.plot(x, y, 'r')

#Heart graph
theta = np.linspace(0, 2*np.pi, 100)
x = 16*np.sin(theta)**3
y = 13*np.cos(theta) - 5*np.cos(2*theta) - 2*np.cos(3*theta) - np.cos(4*theta)

ax = fig.add_subplot(2,3,5)
ax.plot(x, y, 'r')
##################

# Show all graphs
plt.show()	