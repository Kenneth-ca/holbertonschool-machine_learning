#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('HBTNaligned', as_array=True)
triplet_names = load_csv('FVTriplets.csv')
triplets = generate_triplets(images, filenames, triplet_names)


tm = TrainModel('models/face_verification.h5', 0.2)
tm.training_model.summary()
losses = tm.training_model.predict(triplets, batch_size=1)
print(losses.shape)
print(np.mean(losses))
