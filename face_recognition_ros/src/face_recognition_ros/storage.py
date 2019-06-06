import os
from os import path
import bisect

import numpy as np

from face_recognition_ros import encoding
from face_recognition_ros.utils import files


class FaceDatabase():
    """ Class used for the load/storage of previously known faces.

        Our face recongition system collects a set of embeddings
        per each known identity. They can be stored for posterior
        use.

        Attributes:
            MAX_IMGS: Maximum number of samples read from an image directory.
            labels: String descriptors for each identity.
            emb_start: Index at which each identity starts in embedings.
            embeddings: Matrix storing at each row a different embedding.
                To verify each correspondence with an identity, lookup
                the position in emb_start.
    """
    def __init__(self, MAX_IMGS=np.Inf):
        self.MAX_IMGS = MAX_IMGS

        self.labels = []
        self.emb_start = []
        self.embeddings = np.empty((0, 512))  # type: np.ndarray

    def add_identity(self, dir, face_enc, label=None):
        # type: (str, face_encoding.FacialEncoder, str) -> None
        """ Add a new entry to our facial recognition database.

            It will load at most MAX_IMGS and compute each embedding.

            Args:
                dir: Directory containing only images
                label: Identity to store in database.
                    If not present it will use the name of the directory.
        """
        if label is None:
            label = path.basename(dir)

        self.labels.append(label)
        self.emb_start.append(self.embeddings.shape[0])

        images = [path.join(dir, i) for i in os.listdir(dir)]

        n_imgs = min(self.MAX_IMGS, len(images))
        im = encoding.load_images(images[:n_imgs])

        # DONE: Compute embeddings
        emb = face_enc.predict(im)
        self.embeddings = np.vstack((self.embeddings, emb))

    def load(self, dir):
        """ Clear current database and load from files in a directory """
        self.embeddings = np.load(path.join(dir, 'embeddings.npy'),
                                  allow_pickle=False)

        self.emb_start = []
        self.labels = []
        with file(path.join(dir, "id.txt"), mode="r") as f:
            for line in f:  # type: str
                ls = line.split(",")
                self.labels.append(ls[0])
                self.emb_start.append(int(ls[1]))

    def save(self, dir):
        """ Store current database in a directory """
        np.save(path.join(dir, 'embeddings.npy'),
                self.embeddings,
                allow_pickle=False)

        with file(path.join(dir, "id.txt"), mode="w") as f:
            for label, n in zip(self.labels, self.emb_start):
                f.write("{},{}\n".format(label, n))

    def __getitem__(self, item):
        i = bisect.bisect_right(self.emb_start, item)
        if i >= len(self.emb_start) or self.emb_start[i] > item:
            i -= 1
        return self.labels[i], self.embeddings[item, :]

    def __setitem__(self, item, value):
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError
