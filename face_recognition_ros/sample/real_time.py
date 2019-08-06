#!/usr/bin/ python2

import sys
import os
import cv2
import matplotlib.pyplot as plt

from face_recognition_ros.utils import files
sys.path.append(os.path.join(files.PROJECT_ROOT, "../facenet/src"))

from face_recognition_ros import detection
from face_recognition_ros.utils import config


def main():
    cap = cv2.VideoCapture(0)
    #fig, ax = plt.subplots()
    #plt.ion()

    config.load_config()

    det = detection.FacialDetector("dlib")

    while True:
        ret, frame = cap.read()
        if ret:
            print "one"
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            reg = det.extract_images(frame)
            #for r in reg:
                #frame = r.draw(frame)
            # ax.imshow(frame)
            #plt.pause(0.0005)


if __name__ == "__main__":
    main()
