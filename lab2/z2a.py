import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,20,50)
y = np.sin(x)*np.cos(x)
plt.plot(x,y)