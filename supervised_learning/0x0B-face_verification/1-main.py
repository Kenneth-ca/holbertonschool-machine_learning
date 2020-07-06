#!/usr/bin/env python3

from utils import load_csv

triplets = load_csv('FVTriplets.csv')
print(type(triplets), len(triplets))
print(triplets[:10])
