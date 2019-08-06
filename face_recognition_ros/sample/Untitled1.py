#!/usr/bin/env python
# coding: utf-8

import sys
import timeit

# CORE PROJECT
sys.path.append('/home/sam/catkin_ws/src/src/facenet/src')
sys.path.append('/home/sam/catkin_ws/src/src/face_recognition_ros/src')

from face_recognition_ros import detection
from face_recognition_ros.utils import config

config.load_config()
recog = detection.FacialDetector()

IMAGE_PATH = '/home/sam/Pictures/IMG-20190419-WA0001.jpg'
image = cv.imread(IMAGE_PATH, 1)

data = recog.extract_images(image)
print "Done"
