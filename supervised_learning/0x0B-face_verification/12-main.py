#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('HBTNaligned', as_array=True)
triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
triplets = [A[:-2], P[:-2], N[:-2]] # to make all batches divisible by 32

tm = TrainModel('models/face_verification.h5', 0.2)
history = tm.train(triplets, epochs=1)
print(history.history)
