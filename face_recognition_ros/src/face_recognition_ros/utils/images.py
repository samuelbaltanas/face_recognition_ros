# TODO: Rename file or move to extraction
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def plot_bounding_box(
    image, bbs, color=(0, 255, 0), color_enc=None, thickness=10
):
    imbb = image.copy()

    if color_enc is not None:
        imbb = cv.cvtColor(imbb, color_enc)

    for bb in bbs:
        imbb = cv.rectangle(
            imbb, (bb[2], bb[0]), (bb[3], bb[1]), color, thickness=thickness
        )

    plt.imshow(imbb)
    return imbb


def plot_corners(image, bbs, color_enc=None, **kwargs):
    plt.figure()

    if color_enc is not None:
        image = cv.cvtColor(image, color_enc)

    for bb in bbs:
        bb = bb.T
        plt.scatter(bb[0], bb[1], **kwargs)

    plt.imshow(image)
