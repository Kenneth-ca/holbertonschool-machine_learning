#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = np.arange(0, 110, 10)
plt.hist(student_grades, bins=bins, edgecolor='black')
plt.title('Project A')
plt.xlim(0, 100)
plt.xticks(np.arange(0, 110, 10))
plt.xlabel('Grades')
plt.ylim(0, 30)
plt.ylabel('Number of Students')
plt.show()
