#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

# CORE PROJECT
sys.path.append('/home/sam/catkin_ws/src/src/face_recognition_ros/src')

from face_recognition_ros import recognition
from face_recognition_ros.utils import config


# In[5]:


config.load_config()
recog = recognition.Recognition()


# In[12]:


IMAGE_PATH = '/home/sam/Pictures/IMG-20190419-WA0001.jpg'
image = cv.imread(IMAGE_PATH, 1)

data = recog.recognize(image)
