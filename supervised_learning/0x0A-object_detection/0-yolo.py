#!/usr/bin/env python3
"""
Module to perform object detection
"""
import tensorflow.keras as K


class Yolo:
    """
    uses the Yolo v3 algorithm to perform object detection:
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """

        :param self: instance of the class
        :param model_path: is the path to where a Darknet Keras model is stored
        :param classes_path: is the path to where the list of class names
        used for the Darknet model, listed in order of index, can be found
        :param class_t: is a float representing the box score threshold for
        the initial filtering step
        :param nms_t: is a float representing the IOU threshold for non-max
        suppression
        :param anchors: is a numpy.ndarray of shape (outputs, anchor_boxes,
        2) containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the
            Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        :return:
        """
        self.model = K.models.load_model(filepath=model_path)
        with open(classes_path, 'r') as f:
            txt_saved = f.read()
            txt_saved = txt_saved.split('\n')
            if len(txt_saved[-1]) == 0:
                txt_saved = txt_saved[:-1]
        self.class_names = txt_saved
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
