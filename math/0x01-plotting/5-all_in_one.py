#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Figure
fig = plt.figure()
plt.suptitle("All in One")

# 0. Line Graph
axe0 = plt.subplot2grid((3, 2), (0, 0))
axe0.plot(y0, 'r-')
axe0.set_xlim(0, 10)

# 1. Scatter
axe1 = plt.subplot2grid((3, 2), (0, 1))
axe1.scatter(x1, y1, c='m')
axe1.set_title("Men's Height vs Weight", fontsize='x-small')
axe1.set_xlabel('Height (in)', fontsize='x-small')
axe1.set_ylabel('Weight (lbs)', fontsize='x-small')

# 2. Change of scale
axe2 = plt.subplot2grid((3, 2), (1, 0))
axe2.plot(x2, y2)
axe2.set_title("Exponential Decay of C-14", fontsize='x-small')
axe2.set_xlabel('Time (years)', fontsize='x-small')
axe2.set_xlim(left=0, right=28650)
axe2.set_ylabel('Fraction Remaining', fontsize='x-small')
axe2.set_yscale('log')

# 3. Two is better than one
axe3 = plt.subplot2grid((3, 2), (1, 1))
axe3.plot(x3, y31, 'r--', label='C-14')
axe3.plot(x3, y32, 'g-', label='Ra-226')
axe3.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
axe3.set_xlabel('Time (years)', fontsize='x-small')
axe3.set_xlim(left=0, right=20000)
axe3.set_ylim(bottom=0, top=1)
axe3.set_ylabel('Fraction Remaining', fontsize='x-small')
axe3.legend(prop={'size': 'x-small'})

# 4. Frequency
axe4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
bins = np.arange(0, 110, 10)
axe4.hist(student_grades, bins=bins, edgecolor='black')
axe4.set_title('Project A', fontsize='x-small')
axe4.set_xlim(0, 100)
axe4.set_xticks(np.arange(0, 110, 10))
axe4.set_xlabel('Grades', fontsize='x-small')
axe4.set_ylim(0, 30)
axe4.set_ylabel('Number of Students', fontsize='x-small')
plt.tight_layout()
plt.show()
