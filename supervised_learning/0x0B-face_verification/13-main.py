#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('HBTNaligned', as_array=True)
triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
triplets = [A[:-2], P[:-2], N[:-2]]

tm = TrainModel('models/face_verification.h5', 0.2)
tm.train(triplets, epochs=1)
base_model = tm.save('models/trained_fv.h5')
print(base_model is tm.base_model)
print(os.listdir('models'))
