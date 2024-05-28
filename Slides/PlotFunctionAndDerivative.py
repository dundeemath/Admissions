
import numpy as np
import matplotlib.pyplot as plt

dx=0.01
x = np.arange(0, 20, dx)
y = np.sin(x)

dy_dx=np.cos(x)

fig, ax = plt.subplots(1,2)
ax[0].plot(x, y)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y(x)')

ax[1].plot(x, dy_dx)
ax[1].set_xlabel('x')
ax[1].set_ylabel('dy/dx')

ax[0].grid(True)
ax[1].grid(True)

plt.tight_layout()
plt.show()
