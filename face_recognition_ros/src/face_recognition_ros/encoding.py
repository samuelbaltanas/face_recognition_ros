import numpy as np
import tensorflow as tf

from face_recognition_ros.utils import files, config
from face_recognition_ros.third_party import facenet


class FacialEncoder:
    """ Facial detector using Tensorflow and Facenet """

    def __init__(self, conf=None):
        self.session = tf.Session()

        if conf is None:
            conf = config.CONFIG

        # Loading model
        with self.session.as_default():
            facenet.load_model(files.get_model_path("", conf["FACENET"]["model_name"]))

        # Tensors
        def_graph = tf.get_default_graph()
        self._images_placeholder = def_graph.get_tensor_by_name("input:0")
        self._embeddings = def_graph.get_tensor_by_name("embeddings:0")
        self._phase_train_placeholder = def_graph.get_tensor_by_name("phase_train:0")

    def predict(self, face_images):
        images = [preprocess_face(face) for face in face_images]

        feed_dict = {
            self._images_placeholder: images,
            self._phase_train_placeholder: False,
        }

        return self.session.run(self._embeddings, feed_dict=feed_dict)


def prewhiten(img):
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    y = np.multiply(np.subtract(img, mean), 1 / std_adj)
    return y


def preprocess_face(image):
    # Necessary for model 20180402-114759 as explained in:
    # https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset
    image = (np.float32(image) - 127.5) / 128.0
    # image = prewhiten(image)
    return image
