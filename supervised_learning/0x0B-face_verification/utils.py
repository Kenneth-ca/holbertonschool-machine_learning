#!/usr/bin/env python3
"""
Performs many utilities for images
"""
import numpy as np
import glob
import cv2
import csv
import os


def load_images(images_path, as_array=True):
    """
    loads images from a directory or file
    :param images_path: is the path to a directory from which to load images
    :param as_array: is a boolean indicating whether the images should be
    loaded as one numpy.ndarray
    If True, the images should be loaded as a numpy.ndarray of
    shape (m, h, w, c) where:
        m is the number of images
        h, w, and c are the height, width, and number of channels of all
        images, respectively
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
        # IMAGES
        # For Linux
        pic = cv2.imread(image)
        # For Windows - special characters (no need for Linux)
        pic = cv2.imdecode(np.fromfile(image, np.uint8),
                           cv2.IMREAD_UNCHANGED)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pics.append(pic)

        # FILENAMES
        # For Linux
        name = image.split('/')[-1]
        # For Windows - directories separator
        name = name.split('\\')[-1]
        filenames.append(name)

    if as_array is True:
        pics = np.array(pics)

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


def save_images(path, images, filenames):
    """
    saves images to a specific path
    :param path: the path to the directory in which the images should be saved
    :param images: a list/numpy.ndarray of images to save
    :param filenames: a list of filenames of the images to save
    :return: True on success and False on failure
    """
    if os.path.exists(path):
        for i in range(len(images)):
            colored = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path, filenames[i]), colored)
        return True
    else:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    generates triplets
    :param images: numpy.ndarray of shape (n, h, w, 3) containing the various
    images in the dataset
    :param filenames: a list of length n containing the corresponding
    filenames for images
    :param triplet_names: a list of lists where each sublist contains the
    filenames of an anchor, positive, and negative image, respectively
    :return: a list [A, P, N]
        A is a numpy.ndarray of shape (m, h, w, 3) containing the anchor images
        for all m triplets
        P is a numpy.ndarray of shape (m, h, w, 3) containing the positive
        images for all m triplets
        N is a numpy.ndarray of shape (m, h, w, 3) containing the negative
        images for all m triplets
    """
    n, h, w, c = images.shape
    A = []
    P = []
    N = []
    for i in range(len(triplet_names)):
        anchor, positive, negative = triplet_names[i]
        anchor = anchor + ".jpg"
        positive = positive + ".jpg"
        negative = negative + ".jpg"

        if anchor in filenames:
            if positive in filenames:
                if negative in filenames:
                    idx_a = filenames.index(anchor)
                    idx_p = filenames.index(positive)
                    idx_n = filenames.index(negative)

                    A.append(images[idx_a])
                    P.append(images[idx_p])
                    N.append(images[idx_n])
    A = [ele.reshape(1, w, h, c) for ele in A]
    P = [ele.reshape(1, w, h, c) for ele in P]
    N = [ele.reshape(1, w, h, c) for ele in N]

    A = np.concatenate(A)
    P = np.concatenate(P)
    N = np.concatenate(N)
    triplets = [A, P, N]
    return triplets
