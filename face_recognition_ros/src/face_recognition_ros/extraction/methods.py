import numpy as np

from face_recognition_ros.extraction import region
from face_recognition_ros.utils.math import dist, solve_line_intersect


# DONE: Separate different types
# DONE: Extract info of which pose correspond to which box
# TODO: Modify to return EllipseRegion
# TODO: Keypoints are reversed sometimes (RL or LR)
# TODO: Missing side face extraction (1 eye missing)
# DONE: Frontal face (Both sides available)
# DONE: Partial side face (Both eyes available)
# TODO: Set minimum size
# TODO: Output area only within image


def oriented_face_detector(p, confidence_threshold):
    pres = p[:, 2] > confidence_threshold

    bb = None

    if not np.any(pres[[14, 15, 0]]):
        pass
    elif pres[17] and pres[16]:
        # Best condition: two sides present
        ratio = 20.9 / 16.1

        # Side to side vector
        v = (p[17, 0:2] - p[16, 0:2]) / 2
        vt = ratio * v[::-1]
        vt[0] *= -1

        # Corners
        # center = np.mean(p[[0, 16, 17, 14, 15], 0:2], axis=0)
        topl = p[16, 0:2] - vt
        botr = p[17, 0:2] + vt
        topr = p[17, 0:2] - vt
        botl = p[16, 0:2] + vt

        bb = np.float32([topl, topr, botl, botr])
    elif pres[14] and pres[15] and pres[0] and pres[1]:
        if pres[16]:
            orig = p[16, 0:2]
            d = -1
        elif pres[17]:
            orig = p[17, 0:2]
            d = 1
        else:
            return

        rem_idx = np.array([0, 14, 15])
        rem = p[rem_idx, 0:2]

        furth = np.argmax(np.sum((orig - rem) ** 2, axis=1))

        # print(np.sum((p[[pts[0]], 0:2] - p[pts[1:], 0:2])**2, axis=1))
        v = rem[1] - rem[2]
        vs = np.linalg.norm(v)
        v /= vs

        vt = np.copy(v[::-1])
        vt[1] = vt[1] * -1

        upvec = (p[0, 0:2] - p[1, 0:2]) / 2

        a = orig
        b = -1.2 * upvec + rem[0]
        c = rem[furth] + d * v * vs / 2
        d = 1.2 * upvec + rem[0]

        botl = solve_line_intersect(a, b, vt, v)
        botr = solve_line_intersect(b, c, v, vt)
        topr = solve_line_intersect(c, d, vt, v)
        topl = solve_line_intersect(d, a, v, vt)

        bb = np.float32([topl, topr, botl, botr])
    else:
        pass

    return bb


# TODO: Incorporate ideas from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/face/faceDetector.cpp
def oriented_face_detector_2(posePtr, threshold):
    major_axis_radius = 0.0  # Face height
    minor_axis_radius = 0.0  # Face width
    angle = 0.0
    center = np.zeros((2, 1))
    detection_score = np.mean(posePtr[[0, 1, 14, 15, 16, 17], 2])

    neckScoreAbove = posePtr[1, 2] > threshold
    headNoseScoreAbove = posePtr[0, 2] > threshold
    lEarScoreAbove = posePtr[16, 2] > threshold
    rEarScoreAbove = posePtr[17, 2] > threshold
    lEyeScoreAbove = posePtr[14, 2] > threshold
    rEyeScoreAbove = posePtr[15, 2] > threshold

    counter = 0.0

    if lEyeScoreAbove and rEyeScoreAbove:
        # TODO Condition: Both eyes present
        center += (posePtr[14, 0:2] + posePtr[15, 0:2]) / 2.0
        diff = posePtr[15, 0:2] - posePtr[14, 0:2]
        angle += np.arctan2(diff[14, 0], diff[15, 0])
        minor_axis_radius += 3.0 * dist(posePtr[14, 0:2], posePtr[15, 0:2])
        pass

    if lEarScoreAbove and rEarScoreAbove:
        # TODO Condition: Both ears present
        center += (posePtr[16, 0:2] + posePtr[17, 0:2]) / 2.0
        diff = posePtr[17, 0:2] - posePtr[16, 0:2]
        angle += np.arctan2(diff[16, 0], diff[17, 0])
        minor_axis_radius += 2.0 * dist(posePtr[16, 0:2], posePtr[17, 0:2])
        pass

    if neckScoreAbove and headNoseScoreAbove:
        if (
            lEyeScoreAbove == lEarScoreAbove
            and rEyeScoreAbove == rEarScoreAbove
            and lEyeScoreAbove != rEyeScoreAbove
        ):
            # Condition: One side missing
            if lEyeScoreAbove:
                # Condition: Left side only
                center += (posePtr[14, 0:2] + posePtr[16, 0:2] + posePtr[0, 0:2]) / 3.0

                diff = posePtr[16, 0:2] - posePtr[14, 0:2]
                angle += 0.0  # TODO

                width = dist(posePtr[16, 0:2], posePtr[14, 0:2])
                minor_axis_radius += width
                # major_axis_radius += 0.0 * width
            else:
                # Condition: Right side only
                center += (posePtr[15, 0:2] + posePtr[17, 0:2] + posePtr[0, 0:2]) / 3.0

                diff = posePtr[15, 0:2] - posePtr[17, 0:2]
                angle += 0.0  # TODO

                width = dist(posePtr[15, 0:2], posePtr[17, 0:2])
                minor_axis_radius += width
                # major_axis_radius += 0.0 * width
        else:
            # TODO Condition: None or part of both sides present
            pass

        counter += 1.0

    if counter > 0:
        # TODO Fix averaging process
        center /= counter
        minor_axis_radius /= counter
        major_axis_radius /= counter
        angle /= counter

    return region.EllipseRegion(
        major_axis_radius,
        minor_axis_radius,
        angle,
        center[0],
        center[1],
        detection_score,
    )
