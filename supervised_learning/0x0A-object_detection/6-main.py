#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('6-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('../data/yolo')
    boxes = np.array([[119.22100287, 118.62197718, 567.75985556, 440.44121152],
                      [468.53530752, 84.48338278, 696.04923556, 167.98947829],
                      [124.2043716, 220.43365057, 319.4254314 , 542.13706101]])
    box_scores = np.array([0.99537075, 0.91536146, 0.9988506])
    box_classes = np.array([1, 7, 16])
    ind = 0
    for i, name in enumerate(image_paths):
        if "dog.jpg" in name:
            ind = i
            break
    yolo.show_boxes(images[i], boxes, box_classes, box_scores, "dog.jpg")
