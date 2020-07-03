#!/usr/bin/env python3

if __name__ == '__main__':
    import cv2
    import numpy as np
    Yolo = __import__('5-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('../data/yolo')
    pimages, image_shapes = yolo.preprocess_images(images)
    print(type(pimages), pimages.shape)
    print(type(image_shapes), image_shapes.shape)
    i = np.random.randint(0, len(images))
    print(images[i].shape, ':', image_shapes[i])
    cv2.imshow(image_paths[i], pimages[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
