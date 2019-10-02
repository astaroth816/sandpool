import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 1000)
y = np.pi * (1 - np.sin(theta))
graph = plt.subplot(111, polar=True)
graph.plot(theta, y,color='red', linewidth=2)
plt.show()
