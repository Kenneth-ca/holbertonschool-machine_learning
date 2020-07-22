#!/usr/bin/env python3

from triplet_loss import TripletLoss

print(TripletLoss.__bases__)
tl = TripletLoss(0.2)
print(tl.alpha)
print(sorted(tl.__dict__.keys()))
