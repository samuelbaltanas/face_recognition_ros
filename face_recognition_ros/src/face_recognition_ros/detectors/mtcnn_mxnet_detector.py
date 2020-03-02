# coding: utf-8
import typing

import numpy as np

from face_recognition_ros import datum, region

# from face_recognition_ros.detectors import base_face_detector
from face_recognition_ros.third_party import align_mtcnn, mtcnn_mxnet
from face_recognition_ros.utils import files


class MtcnnMxnetDetector:
    def __init__(self, conf):
        self.detector = mtcnn_mxnet.MtcnnDetector(
            model_folder=files.PROJECT_ROOT + "/data/models/mtcnn-mxnet"
        )

    def predict(
        self, X: np.ndarray, threshold=0.95, extract_image=False, align=True
    ) -> typing.List[datum.Datum]:
        ret = self.detector.detect_face(X)

        if ret is None:
            return []

        bbox, points = ret
        if bbox.shape[0] == 0:
            return []

        data = []
        for bb, point in zip(*ret):

            if bb[4] >= threshold:
                dat = datum.Datum(
                    region=region.BoundingBox(bb[:4], bb[4]), keypoints=point,
                )

                if extract_image:
                    if align:
                        point = point.reshape((2, 5)).T
                    elif extract_image:
                        point = None

                    aligned = align_mtcnn.align(X, bb, point, image_size="112,112")
                    dat.image = aligned

                data.append(dat)

        return data
