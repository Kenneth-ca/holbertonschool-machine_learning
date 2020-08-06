#!/usr/bin/env python3

posterior = __import__('100-continuous').posterior
import numpy as np

print(posterior(7, 50, 0.1, 0.15))
print(posterior(7, 50, 0.8, 0.9))
print(posterior(26, 31, 0.8, 0.9))
print(posterior(26, 31, 0.1, 0.15))
