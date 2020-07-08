#!/usr/bin/env python3

import numpy as np
import re
from train_model import TrainModel
from utils import load_images


images, filenames = load_images('HBTNaligned', as_array=True)
identities = [re.sub('[0-9]', '', f[:-4]) for f in filenames]
print(set(identities))
thresholds = np.linspace(0.01, 0.1, 100)
tm = TrainModel('models/face_verification.h5', 0.2)
tau, f1, acc = tm.best_tau(images, identities, thresholds)
print(tau)
print(f1)
print(acc)
