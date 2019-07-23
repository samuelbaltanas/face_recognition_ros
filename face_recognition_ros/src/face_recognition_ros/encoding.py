import sys
from os import path

import tensorflow as tf

from face_recognition_ros.utils import files, config


sys.path.append(path.join(files.PROJECT_ROOT, "../facenet/src"))
import facenet  # noqa: E402


class FacialEncoder:
    """ Facial detector using Tensorflow and Facenet """

    def __init__(self, conf=None):
        self.session = tf.Session()

        if conf is None:
            conf = config.CONFIG

        # Loading model
        with self.session.as_default():
            facenet.load_model(files.get_model_path(conf["FACENET"]["model_name"]))

        # Tensors
        def_graph = tf.get_default_graph()
        self._images_placeholder = def_graph.get_tensor_by_name("input:0")
        self._embeddings = def_graph.get_tensor_by_name("embeddings:0")
        self._phase_train_placeholder = def_graph.get_tensor_by_name("phase_train:0")

    def predict(self, face_images):
        # images = map(
        #    lambda x: image_preprocessing.preprocess_face(x[1]), images
        # )

        feed_dict = {
            self._images_placeholder: face_images,
            self._phase_train_placeholder: False,
        }

        return self.session.run(self._embeddings, feed_dict=feed_dict)
