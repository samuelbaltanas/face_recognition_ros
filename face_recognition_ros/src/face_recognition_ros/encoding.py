import sys
from os import path

import tensorflow as tf

from face_recognition_ros.utils import files

sys.path.append(path.join(files.PROJECT_ROOT, "../facenet/src"))
import compare
import facenet


class FacialEncoder:
    """ Facial detector using Tensorflow and Facenet
CONFIG_PATH + cfg
    """

    def __init__(self, session, config):
        self.session = session

        # Loading model
        facenet.load_model(files.get_model_path(config["model_name"]))

        # Tensors
        def_graph = tf.get_default_graph()
        self._images_placeholder = def_graph.get_tensor_by_name("input:0")
        self._embeddings = def_graph.get_tensor_by_name("embeddings:0")
        self._phase_train_placeholder = def_graph.get_tensor_by_name(
            "phase_train:0"
        )

    def predict(self, images):
        feed_dict = {
            self._images_placeholder: images,
            self._phase_train_placeholder: False,
        }
        return self.session.run(self._embeddings, feed_dict=feed_dict)


def load_images(image_files, image_size=160, margin=44, gpu_mem_fraction=1.0):
    # type: (str, int, int, float) -> Any
    """ Loads and aligns the images from file

        Args:
            image_files: Path of the images to compare
            image_size: Size of the images
            margin:
            gpu_mem_fraction:

        Return:
            Array of loaded images
    """
    return compare.load_and_align_data(
        image_files, image_size, margin, gpu_mem_fraction
    )
