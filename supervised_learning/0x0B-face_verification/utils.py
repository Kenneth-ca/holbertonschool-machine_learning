#!/usr/bin/env python3
"""
Performs many utilities for images
"""
import numpy as np
import glob
import cv2
import csv


def load_images(images_path, as_array=True):
    """
    loads images from a directory or file
    :param images_path: is the path to a directory from which to load images
    :param as_array: is a boolean indicating whether the images should be
    loaded as one numpy.ndarray
    If True, the images should be loaded as a numpy.ndarray of
    shape (m, h, w, c) where:
        m is the number of images
        h, w, and c are the height, width, and number of channels of all images,
         respectively
        If False, the images should be loaded as a list of individual
        numpy.ndarrays
    :return: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated with each image in
        images
    """
    paths = glob.glob(images_path + "/*", recursive=False)
    paths.sort()

    pics = []
    filenames = []

    for image in paths:
        pic = cv2.imread(image)
        # For Windows - special characters (no need for Linux)
        #pic = cv2.imdecode(np.fromfile(image, np.uint8),
        #                   cv2.IMREAD_UNCHANGED)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pics.append(pic)
        # For Linux
        filenames.append(image.split('/')[-1])
        # For Windows - directories separator
        # filenames.append(image.split('\\')[-1])

    if as_array is True:
        filenames = np.array(filenames)

    return (pics, filenames)

def load_csv(csv_path, params={}):
    """
    loads the contents of a csv file as a list of lists
    :param csv_path: is the path to the csv to load
    :param params: are the parameters to load the csv with
    :return: a list of lists representing the contents found in csv_path
    """
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, params)
        list_csv = [row for row in csv_reader]
    return list_csv
