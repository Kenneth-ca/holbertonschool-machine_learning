#!/usr/bin/env python3

posterior = __import__('100-continuous').posterior
import numpy as np

print(posterior(0, 37, 0.0, 0.1))
print(posterior(109, 109, 0.95, 1.0))
