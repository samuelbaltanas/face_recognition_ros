import tensorflow as tf

from face_recognition_ros import region
from face_recognition_ros.detectors import base_face_detector
from face_recognition_ros.third_party import align_mtcnn, mtcnn_tensorflow
from face_recognition_ros.utils import files


class MtcnnFaceDetector(base_face_detector.BaseFaceDetector):

    min_size = 20
    threshold = [0.6, 0.6, 0.7]
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
            self.pnet, self.rnet, self.onet = mtcnn_tensorflow.create_mtcnn(
                sess, files.PROJECT_ROOT + "/data/models/mtcnn_tensorflow"
            )

    def extract_region(
        self, image, threshold=0.9,  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ):

        bbs, points = mtcnn_tensorflow.detect_face(
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

        return regions, (bbs, points)

    def extract_images(self, image, regions=None, raw_detection=None, align=False):
        if raw_detection is None:
            regions, raw_detection = self.extract_region(image, 0)

        if not align:
            return super(MtcnnFaceDetector, self).extract_images(
                image, regions, raw_detection
            )
        else:
            bbox, points = raw_detection

            res = []
            for idx, box in enumerate(bbox):
                point = points[:, idx].reshape((2, 5)).T
                aligned = align_mtcnn.align(image, box, point, image_size="160,160")
                res.append(aligned)

            return res
