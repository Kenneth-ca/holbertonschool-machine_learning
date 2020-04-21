#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

names = ['Farrah', 'Fred', 'Felicia']
fs = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

for i in range(len(fruit)):
    plt.bar(names, fruit[i], bottom=np.sum(fruit[:i], axis=0),
            color=colors[i], label=fs[i], width=0.5)

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.yticks(np.arange(0, 90, 10))
plt.legend()
plt.show()
