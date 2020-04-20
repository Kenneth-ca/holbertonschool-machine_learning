#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, 'r--', label='C-14')
plt.plot(x, y2, 'g-', label='Ra-226')
plt.title('Exponential Decay of Radioactive Elements')
plt.xlabel('Time (years)')
plt.xlim(left=0, right=20000)
plt.ylim(bottom=0, top=1)
plt.ylabel('Fraction Remaining')
plt.legend()
plt.show()
