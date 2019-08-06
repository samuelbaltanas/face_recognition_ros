import logging
import os
import bisect

import numpy as np
import cv2

from face_recognition_ros import encoding, detection
from face_recognition_ros.utils import files, config
from face_recognition_ros.utils.math import dist


class FaceMatcher:
    def __init__(self, conf=None):
        # type: (dict) -> None
        self.database = FaceDatabase()
        if conf is None:
            conf = config.CONFIG
        storage_config = conf["STORAGE"]

        if storage_config["database_folder"] == "":
            data_path = files.get_face_database(storage_config["database_name"])
        else:
            data_path = os.path.join(
                storage_config["database_folder"], storage_config["database_name"]
            )
        self.database.load(data_path)

    def recognize1(self, embeding, threshold=1.1):
        # TODO: Decide method of comparison. Maybe use a SVM or KNN to decide
        # type: (np.ndarray) -> (str, float)

        res = []
        for label, embs in iter(self.database):
            # d = np.sqrt(np.sum((embs - embeding) ** 2, axis=1))  # type: np.ndarray
            d = dist(embs, embeding, func=0)  # type: np.ndarray

            m = np.mean(d <= threshold)

            if m > 0:
                logging.debug("Hypothesis [{}] accepted. Distance={}".format(label, m))
                res.append((label, m))
            else:
                logging.debug("Hypothesis [{}] discarded. Distance={}".format(label, m))

        if not res:
            logging.debug("Unknown face. Distance={}".format(np.NaN))
            return "", np.NaN
        else:
            label, d = max(res, key=lambda x: x[1])
            logging.info("Id: {} recognized with dist {}".format(label, d))
            return label, d

    def recognize(self, embeding, threshold=1.0):
        # type: (np.ndarray) -> (str, float)
        d_list = dist(self.database.embeddings, embeding, func=1)  # type: np.ndarray

        idx = d_list.argmin(axis=0)
        d = d_list[idx]

        if d_list[idx] < threshold:
            iden = self.database.get_pair(idx)[0]
            logging.info("Id: {} recognized with dist {}".format(iden, d))
            return iden, d
        else:
            logging.debug("Unknown face. Distance={}".format(d))
            return "", d


class FaceDatabase:
    """ Class used for the load/storage of previously known faces.

        Our face recongition system collects a set of embeddings
        per each known identity. They can be stored for posterior
        use.

        Attributes:
            labels: String descriptors for each identity.
            emb_start: Index at which each identity starts in embedings.
            embeddings: Matrix storing at each row a different embedding.
                To verify each correspondence with an identity, lookup
                the position in emb_start.
    """

    def __init__(self):
        self.labels = []
        self.emb_start = []
        self.embeddings = np.empty((0, 512))  # type: np.ndarray

    def add_identity(self, label, embeddings):
        start = self.embeddings.shape[0]

        self.labels.append(label)
        self.emb_start.append(start)
        self.embeddings = np.vstack((self.embeddings, embeddings))

    def get_pair(self, idx):
        i = bisect.bisect_right(self.emb_start, idx)
        if i >= len(self.emb_start) or self.emb_start[i] > idx:
            i -= 1
        return self.labels[i], self.embeddings[idx, :]

    def __getitem__(self, item):
        if item >= len(self.labels):
            raise IndexError()
        if item == len(self.labels) - 1:
            return self.embeddings[self.emb_start[item] :, :]
        else:
            return self.embeddings[self.emb_start[item] : self.emb_start[item + 1], :]

    def __setitem__(self, item, value):
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        for idx, label in enumerate(self.labels):
            yield label, self[idx]

    def load(self, dir):
        """ Clear current database and load from files in a directory """
        db_file = os.path.join(dir, "embeddings.npy")
        logging.info("Loading database from {}".format(db_file))
        self.embeddings = np.load(db_file, allow_pickle=False)

        self.emb_start = []
        self.labels = []
        with open(os.path.join(dir, "id.txt"), "r") as f:
            for line in f:  # type: str
                ls = line.split(",")
                self.labels.append(ls[0])
                self.emb_start.append(int(ls[1]))

        logging.debug("Database loaded: \n{}".format(str(self)))

    def save(self, dir):
        """ Store current database in a directory """
        np.save(
            os.path.join(dir, "embeddings.npy"), self.embeddings, allow_pickle=False
        )

        with open(os.path.join(dir, "id.txt"), "w") as f:
            for label, n in zip(self.labels, self.emb_start):
                f.write("{},{}\n".format(label, n))

    def __str__(self):
        contents = []
        length = len(self.labels)
        for idx, name in enumerate(self.labels):
            if idx == length - 1:
                num = length - self.emb_start[idx]
            else:
                num = self.emb_start[idx + 1] - self.emb_start[idx]

            contents.append("{} {}".format(name, num))

        return "\n".join(contents)


def create_dataset(image_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    config.load_config()
    config.logger_config()

    det = detection.FacialDetector()
    enc = encoding.FacialEncoder()

    database = FaceDatabase()

    for label, file_list in files.image_folder_traversal(image_dir):
        emb_list = []
        logging.info("Adding [{}] to database".format(label))
        for file_name in file_list:
            image = cv2.imread(file_name)
            if image is None:
                continue

            try:
                faces = det.extract_images(image)

                embedding = enc.predict(faces)
                emb_list.append(embedding)
                logging.debug("Image added: {}".format(file_name))
            except Exception as e:
                logging.warn("Error in image {}".format(file_name))
                logging.warn(e.message)
                os.remove(file_name)
                logging.warn("File {} deleted.".format(file_name))

        logging.info(
            "Added identity: [{}]. Num. of images: {}".format(label, len(emb_list))
        )
        database.add_identity(label, np.vstack(emb_list))

    database.save(out_dir)


if __name__ == "__main__":
    create_dataset(
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_dataset",
        "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_default",
    )
