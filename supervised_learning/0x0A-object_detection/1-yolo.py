#!/usr/bin/env python3
"""
Module to perform object detection
"""
import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """

        :param outputs: is a list of numpy.ndarrays containing the
        predictions from the Darknet model for a single image
        :param image_size: is a numpy.ndarray containing the image’s original
        size [image_height, image_width]
        :return: a tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidence = []
        box_class_probs = []
        img_h, img_w = image_size
        for index, out in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = out.shape
            # Boxes inside
            box = np.zeros(out[:, :, :, :4].shape)
            # Center coordinates, width and height of the output
            t_x = out[:, :, :, 0]
            t_y = out[:, :, :, 1]
            t_w = out[:, :, :, 2]
            t_h = out[:, :, :, 3]

            # Width and height of the predefined anchor
            pw_total = self.anchors[:, :, 0]
            pw = np.tile(pw_total[index], grid_width)
            pw = pw.reshape(grid_width, 1, len(pw_total[index]))
            ph_total = self.anchors[:, :, 1]
            ph = np.tile(ph_total[index], grid_height)
            ph = ph.reshape(grid_height, 1, len(ph_total[index]))

            # Corners of the grid
            cx = np.tile(np.arange(grid_width), grid_height)
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.tile(np.arange(grid_width), grid_height)
            cy = cy.reshape(grid_height, grid_height).T
            cy = cy.reshape(grid_height, grid_height, 1)

            # Boxes predictions
            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy
            bw = np.exp(t_w) * pw
            bh = np.exp(t_h) * ph

            # Normalizing
            bx = bx / grid_width
            by = by / grid_height
            bw = bw / self.model.input.shape[1].value
            bh = bh / self.model.input.shape[2].value

            # Coordinates
            # Top left (x1, y1)
            # Bottom right (x2, y2)
            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0]

            # Append boxes
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            # Box confidence
            aux = out[:, :, :, 4]
            conf = (1 / (1 + np.exp(-aux)))
            conf = conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_confidence.append(conf)

            # Box class probabilities
            aux = out[:, :, :, 5:]
            probs = (1 / (1 + np.exp(-aux)))
            box_class_probs.append(probs)

        return (boxes, box_confidence, box_class_probs)
