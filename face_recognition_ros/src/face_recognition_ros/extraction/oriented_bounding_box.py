""" Face extraction arbitrarily oriented bounding boxes.

    It uses the position of the eyes or the ears for alignment.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# TODO: Delete

def extract_from_pose(datum, confidence_threshold=0.00):
    bbs = []
    for idx, p in enumerate(datum.poseKeypoints):
        bb = extraction_method_1(p, confidence_threshold)
        if bb is not None:
            bbs.append((idx, bb))

    return bbs


def plot_bounding_boxes(image, bbs, color_enc=None, **kwargs):
    plt.figure()

    if color_enc is not None:
        image = cv2.cvtColor(image, color_enc)

    for _, bb in bbs:
        bb = bb.T
        plt.scatter(bb[0], bb[1], **kwargs)

    plt.imshow(image)
