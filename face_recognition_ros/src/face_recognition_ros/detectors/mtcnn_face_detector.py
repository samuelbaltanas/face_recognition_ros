import tensorflow as tf

from align import detect_face

from face_recognition_ros.extraction import region


class MtcnnFaceDetector:

    min_size = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    def __init__(self):
        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

    def extract_region(self, image):
        regions = []

        bbs, _ = detect_face.detect_face(
            image,
            self.min_size,
            self.pnet,
            self.rnet,
            self.onet,
            self.threshold,
            self.factor,
        )

        for bb in bbs:
            r = region.RectangleRegion(
                bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], bb[4]
            )
            regions.append(r)

        return regions
