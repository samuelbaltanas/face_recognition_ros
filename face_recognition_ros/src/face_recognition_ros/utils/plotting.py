import numpy as np
from face_recognition_ros.datum import Datum


def draw_all_detections(image, data):
    # type: (np.ndarray, list[Datum]) -> np.ndarray
    im_cpy = image.copy()
    for i in data:
        im_cpy = i.draw(im_cpy)

    return im_cpy
