#!/usr/bin/env python3
"""
Create the class FaceAlign
"""
import dlib
import cv2
import numpy as np


class FaceAlign:
    """
    Detects faces and align them
    """

    def __init__(self, shape_predictor_path):
        """
        class constructor
        :param shape_predictor_path: is the path to the dlib shape predictor
        model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        detects a face in an image
        :param image: is a numpy.ndarray of rank 3 containing an image from
        which to detect a face
        :return: a dlib.rectangle containing the boundary box for the face in
        the image, or None on failure
        """
        try:
            gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

            # With 1 as argument will use upsampling to detect more faces
            faces = self.detector(gray, 1)
            area = 0
            rect = dlib.rectangle(0, 0, image.shape[1], image.shape[0])

            for face in faces:
                if face.area() > area:
                    area = face.area()
                    rect = face

            return rect
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        finds facial landmarks
        :param image: is a numpy.ndarray of an image from which to find
        facial landmarks
        :param detection: is a dlib.rectangle containing the boundary box of
        the face in the image
        :return: a numpy.ndarray of shape (p, 2)containing the landmark
        points, or None on failure
            p is the number of landmark points
            2 is the x and y coordinates of the point
        """
        try:
            gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
            predictor = self.shape_predictor(gray, detection)
            num_landmarks = predictor.num_parts

            np_landmarks = np.zeros((num_landmarks, 2), dtype="int")
            for i in range(0, num_landmarks):
                np_landmarks[i, 0] = predictor.part(i).x
                np_landmarks[i, 1] = predictor.part(i).y

            return np_landmarks
        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        aligns an image for face verification
        :param image: a numpy.ndarray of rank 3 containing the image to be
        aligned
        :param landmark_indices: numpy.ndarray of shape (3,) containing the
        indices of the three landmark points that should be used for the
        affine transformation
        :param anchor_points: numpy.ndarray of shape (3, 2) containing the
        destination points for the affine transformation, scaled to the
        range [0, 1]
        :param size: is the desired size of the aligned image
        :return: a numpy.ndarray of shape (size, size, 3) containing the
        aligned image, or None if no face is detected
        """
        try:
            detect = self.detect(image)
            landmark = self.find_landmarks(image, detect)

            srcTri = landmark[landmark_indices]
            srcTri = srcTri.astype(np.float32)
            # destination
            anchor = anchor_points * size

            warp_mat = cv2.getAffineTransform(srcTri, anchor)

            warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

            return warp_dst
        except RuntimeError:
            return None
