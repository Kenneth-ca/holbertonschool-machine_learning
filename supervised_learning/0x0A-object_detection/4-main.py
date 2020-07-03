#!/usr/bin/env python3

if __name__ == '__main__':
    import cv2
    import numpy as np
    Yolo = __import__('4-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('../data/yolo')
    i = np.random.randint(0, len(images))
    cv2.imshow(image_paths[i], images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
