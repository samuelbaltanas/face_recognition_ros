#!/usr/bin/env python2
import cv2 as cv


def capture_from_cam():
    cap = cv.VideoCapture(0)
    _, frame = cap.read()
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('frame',gray)
    cap.release()
    # cv.destroyAllWindows()
    return frame
