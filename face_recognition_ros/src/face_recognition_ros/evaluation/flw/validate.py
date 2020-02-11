import pickle

import cv2
import numpy as np
from scipy import interpolate, optimize
from sklearn import metrics

from face_recognition_ros import encoding, encoding_arc
from face_recognition_ros.evaluation.flw import lfw_utils
from face_recognition_ros.utils import config

LFW_ROOT = "/home/sam/datasets/flw/"


def main(
    flw_dir=LFW_ROOT + "flw_mtcnnpy_160",
    pairs_filename=LFW_ROOT + "pairs.txt",
    store_file="/home/sam/Desktop/face.pkl",
    store=1,
):
    if store == 2:
        with open(store_file) as f:
            embedding_list, y_true = pickle.load(f)
            y_true = y_true.flatten()
    else:
        config.load_config()

        encoder = encoding.FacialEncoder()
        # encoder = encoding_arc.EncodingArc()

        embedding_list = []
        y_true = []

        for (path0, path1), issame in lfw_utils.get_paths(
            flw_dir, pairs_filename
        ):

            im0 = cv2.imread(path0)
            im1 = cv2.imread(path1)

            embs = encoder.predict([im0, im1])

            embedding_list.append(embs)
            y_true.append(issame)

        embedding_list = np.vstack(embedding_list)
        y_true = np.array(y_true)

        if store == 1:
            with open(store_file, "wb") as f:
                pickle.dump((embedding_list, y_true), f)

    tpr, fpr, accuracy, val, val_std, far = lfw_utils.evaluate(
        embedding_list, y_true
    )

    print("Accuracy: %2.5f+-%2.5f" % (np.mean(accuracy), np.std(accuracy)))
    print("Validation rate: %2.5f+-%2.5f @ FAR=%2.5f" % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print("Area Under Curve (AUC): %1.3f" % auc)
    eer = optimize.brentq(
        lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0
    )
    print("Equal Error Rate (EER): %1.3f" % eer)


if __name__ == "__main__":
    main()
