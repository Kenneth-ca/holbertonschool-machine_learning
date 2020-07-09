#!/usr/bin/env python3

import numpy as np
from verification import FaceVerification

np.random.seed(0)
database = np.random.randn(5, 128)
identities = ['Holberton', 'school', 'is', 'the', 'best!']
fv = FaceVerification('models/trained_fv.h5', database, identities)
fv.model.summary()
print(fv.database)
print(fv.identities)
