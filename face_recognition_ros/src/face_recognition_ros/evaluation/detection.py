#!/usr/bin/env python2

from __future__ import with_statement, print_function

import logging
from os import path

fold_dir = "/home/sam/Desktop/Workspace/uni/TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/FDDB-folds/"
image_dir = "/home/sam/Desktop/Workspace/uni/TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/datasets/originalPic"
out_dir = "/home/sam/Desktop/"


def main(fold_dir=fold_dir, image_dir=image_dir, out_dir=out_dir):
    for i, _ in fddb_traversal(fold_dir, image_dir, out_dir):
        pass


def fddb_traversal(fold_dir, image_dir, out_dir):
    # Traverse 10 fold files
    for i in range(1, 11):
        fl = "FDDB-fold-{:02d}.txt".format(i)
        fold_file = path.join(fold_dir, fl)
        # 1 results file per fold
        out_file = path.join(out_dir, fl)
        with open(fold_file, "r") as fold:
            logging.debug("Output: {}".format(out_file))
            with open(out_file, "w") as out_fd:
                logging.debug("Fold file: {}".format(fold_file))
                # Traverse over files
                for line in fold:
                    img_file = path.join(image_dir, line)
                    logging.debug("Image file: {}".format(img_file))
                    yield img_file, out_fd
    logging.info("Traversal over FDDB dataset finished.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    main()
