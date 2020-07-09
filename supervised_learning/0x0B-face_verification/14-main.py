#!/usr/bin/env python3

import numpy as np
from train_model import TrainModel

tm = TrainModel('models/face_verification.h5', 0.2)

np.random.seed(0)
y_true = np.random.randint(0, 2, 10)
y_pred = np.random.randint(0, 2, 10)
print(tm.f1_score(y_true, y_pred))
print(tm.accuracy(y_true, y_pred))
