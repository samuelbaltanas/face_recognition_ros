"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import random
from time import sleep

import facenet

from face_recognition_ros import detection
from face_recognition_ros.utils import config

config.load_config()


def create_dataset_mtcnn(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, " ".join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print("Loading detector")

    # Create and choose detector
    detector = detection.FacialDetector(args.method)

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(
        output_dir, "bounding_boxes_%05d.txt" % random_key
    )

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(
                    output_class_dir, filename + ".png"
                )
                print(image_path)
                if not os.path.exists(output_filename):
                    # Load images
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = "{}: {}".format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write("%s\n" % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        regions, raw_detection = detector.extract_region(img)

                        nrof_faces = len(regions)
                        if nrof_faces > 0:
                            img_size = np.asarray(img.shape)[0:2]
                            if (
                                nrof_faces > 1
                                and not args.detect_multiple_faces
                            ):
                                bounding_box_size = np.array(
                                    [r.size() for r in regions]
                                )
                                img_center = img_size / 2
                                offsets = [
                                    r.offset(img_center).flatten()
                                    for r in regions
                                ]
                                offset_dist_squared = np.sum(
                                    np.power(offsets, 2.0), 1
                                )
                                index = np.argmax(
                                    bounding_box_size
                                    - offset_dist_squared * 2.0
                                )  # some extra weight on the centering
                                regions = [regions[index]]
                                raw_detection = [raw_detection[index]]

                            # Extract images
                            for i, (reg, raw) in enumerate(
                                zip(regions, raw_detection)
                            ):
                                scaled = detector.extract_images(
                                    img, regions=[reg], raw_detection=[raw]
                                )[0]

                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(
                                    output_filename
                                )
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(
                                        filename_base, i, file_extension
                                    )
                                else:
                                    output_filename_n = "{}{}".format(
                                        filename_base, file_extension
                                    )
                                misc.imsave(output_filename_n, scaled)
                                text_file.write("%s\n" % (repr(reg)))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write("%s\n" % (output_filename))

    print("Total number of images: %d" % nrof_images_total)
    print(
        "Number of successfully aligned images: %d" % nrof_successfully_aligned
    )


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=detection.METHODS.keys(),
        help="Choose detection method",
        default="opencv",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory with unaligned images.",
        default="~/datasets/flw/raw/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory with aligned face thumbnails.",
        default="~/datasets/flw/flw_opencv_160/",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Image size (height, width) in pixels.",
        default=182,
    )
    parser.add_argument(
        "--margin",
        type=int,
        help="Margin for the crop around the bounding box (height, width) in pixels.",
        default=44,
    )
    parser.add_argument(
        "--random_order",
        help="Shuffles the order of images to enable alignment using multiple processes.",
        action="store_true",
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        help="Upper bound on the amount of GPU memory that will be used by the process.",
        default=1.0,
    )
    parser.add_argument(
        "--detect_multiple_faces",
        type=bool,
        help="Detect and align multiple faces per image.",
        default=False,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    create_dataset_mtcnn(parse_arguments(sys.argv[1:]))


def get_flw_sample_path(flw_path, person, img=None):
    if not isinstance(person, list):
        person = [person]

    if img is None:
        img = [0] * len(person)
    elif not isinstance(img, list):
        img = [img]

    flw_dir = os.listdir(flw_path)
    person_path = [os.path.join(flw_path, flw_dir[i]) for i in person]

    person_img = [
        os.path.join(p, os.listdir(p)[i]) for p, i in zip(person_path, img)
    ]

    return person_img
