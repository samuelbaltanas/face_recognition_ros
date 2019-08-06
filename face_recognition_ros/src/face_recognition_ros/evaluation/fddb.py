#!/usr/bin/env python2

from __future__ import with_statement, print_function

import errno
import os
from os import path
import logging
import typing

import cv2

from face_recognition_ros import detection
from face_recognition_ros.core import region
from face_recognition_ros.utils import config, files

fold_dir = path.expanduser("~/datasets/fddb/FDDB-folds/")
image_dir = path.expanduser("~/datasets/fddb/originalPics/")
out_dir = path.join(files.PROJECT_ROOT, "data/eval/mtcnn_params_detection_default")
sol_dir = path.expanduser("~/datasets/fddb/FDDB-folds/")

config.logger_config()
config.load_config()
log = logging.getLogger("face_recognition_ros")


def main(
    detection_method="mtcnn", fold_dir=fold_dir, image_dir=image_dir, out_file=out_dir
):
    log.setLevel(logging.DEBUG)
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    detector = detection.FacialDetector(detection_method)
    for im_file, out_fd in fddb_traversal(fold_dir, image_dir, out_dir):
        image = cv2.imread(im_file)
        if image is None:
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), im_file)

        faces, _ = detector.extract_region(image)
        out_fd.write("{}\n".format(len(faces)))

        for face in faces:
            out_fd.write("{}\n".format(repr(face)))


def expand_image_path(image_dir, image_file):
    im_file = path.join(image_dir, image_file)
    im_file = im_file[:-1]
    im_file = "".join([im_file, ".jpg"])
    return im_file


def fddb_solution(sol_dir=sol_dir, image_dir=image_dir):
    # type: (str, str) -> (str, list)
    # Traverse 10 fold files
    counter = 0
    for i in range(1, 11):
        fl = "FDDB-fold-{:02d}-ellipseList.txt".format(i)
        fold_file = path.join(sol_dir, fl)
        # 1 results file per fold
        with open(fold_file, "r") as fold:
            # Traverse over files
            while True:
                im_path = fold.readline()
                if not im_path:
                    break
                im_path = expand_image_path(image_dir, im_path)
                num = int(fold.readline())
                res = []
                for i in range(num):
                    res.append(parse_region_line(fold.readline()))
                log.debug("Progression: {}/2844".format(counter))
                log.debug("Image file: file://{}".format(im_path))
                yield im_path, res
                counter += 1


def parse_region_line(line, type=2):
    contents = line[:-1].split()
    if type == 1:
        # Rectangle
        dat = region.RectangleRegion(
            float(contents[0]),
            float(contents[1]),
            float(contents[2]),
            float(contents[3]),
            float(contents[4]),
        )
    elif type == 2:
        # Ellipse
        dat = region.EllipseRegion(
            float(contents[0]),
            float(contents[1]),
            float(contents[2]),
            float(contents[3]),
            float(contents[4]),
            float(contents[5]),
        )
    else:
        raise ValueError()
    return dat


def fddb_traversal(fold_dir, image_dir, out_dir):
    # type: (str, str, str) -> (str, typing.TextIO)
    # Traverse 10 fold files
    counter = 0
    for i in range(1, 11):
        fl = "FDDB-fold-{:02d}.txt".format(i)
        fl_out = "FDDB-fold-{:02d}-out.txt".format(i)
        fold_file = path.join(fold_dir, fl)
        # 1 results file per fold
        out_file = path.join(out_dir, fl_out)
        with open(fold_file, "r") as fold:
            log.debug("Output: {}".format(out_file))
            with open(out_file, "w") as out_fd:
                log.debug("Fold file: {}".format(fold_file))
                # Traverse over files
                for line in fold:
                    out_fd.write(line)
                    img_file = expand_image_path(image_dir, line)
                    log.debug("Progression: {}/~2800".format(counter))
                    log.debug("Image file: file://{}".format(img_file))
                    yield img_file, out_fd
                    counter += 1
    log.info("Traversal over FDDB dataset finished.")


if __name__ == "__main__":
    main()
