import numpy as np
import cv2


def extract_bounding_box_image(image, bbs):
    faces = []

    for bb in bbs:
        # TODO: Fix black options
        face = image[bb[0] : bb[1], bb[2] : bb[3]]
        face = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)

    return faces


def extract_corners_image(image, bbs, shape=(160, 160)):
    faces = []

    for bb in bbs:
        fitted = np.float32(
            [[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]]
        )
        M = cv2.getPerspectiveTransform(bb, fitted)
        fac = cv2.warpPerspective(image, M, shape)
        faces.append(fac)

    return faces


"""
def extract_corners_homograpy(image, bbs, shape=(160, 160)):
    faces = []

    outh = np.float32([[0, 0], [shape[0], 0],
                     [0, shape[1]], [shape[0], shape[1]]])
    outh = cv2.convertPointsToHomogeneous(outh)

    for bb in bbs:
        inh = cv2.convertPointsToHomogeneous(bb)

        H = cv2.findHomography(inh, outh)[0]
        fac = cv2.warpPerspective(image, H, shape)
        faces.append(fac)

    return faces
"""
