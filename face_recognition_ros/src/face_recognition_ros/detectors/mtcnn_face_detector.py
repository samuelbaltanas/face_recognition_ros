import tensorflow as tf

from align import detect_face

from face_recognition_ros.core import region
from face_recognition_ros.detectors import base_face_detector


class MtcnnFaceDetector(base_face_detector.BaseFaceDetector):

    min_size = 50
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    def __init__(self, conf):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=conf["gpu_mem_fraction"]
            )
            sess = tf.Session(
                config=tf.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False
                )
            )
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

    def extract_region(self, image, threshold=0.8):

        bbs, _ = detect_face.detect_face(
            image,
            self.min_size,
            self.pnet,
            self.rnet,
            self.onet,
            self.threshold,
            self.factor,
        )

        regions = [
            region.RectangleRegion(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], bb[4])
            for bb in bbs
            if bb[4] >= threshold
        ]

        return regions, bbs
