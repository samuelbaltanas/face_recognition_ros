#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
# get_ipython().magic(u'matplotlib inline')

import cv2
import sys
import os
import logging
import itertools

import matplotlib.pyplot as plt

sys.path.append("/home/sam/catkin_ws/src/src/face_recognition_ros/src")

from face_recognition_ros import encoding, encoding_arc, detection
from face_recognition_ros.core import datum
from face_recognition_ros.classifiers import default, svm, knn
from face_recognition_ros.utils import config


# In[2]:


IMAGE_PATH = "/home/sam/Pictures/IMG-20190419-WA0001.jpg"
image = cv2.cvtColor(cv2.imread(IMAGE_PATH, 1), cv2.COLOR_BGR2RGB)

RESOLUTION = (1280, 720)
# RESOLUTION = (640, 480)
METHOD = "mtcnn"
LOOPS = 10

image = cv2.resize(image, RESOLUTION)
# plt.imshow(image)


# In[3]:


conf = config.load_config()

detector = detection.FacialDetector(method=METHOD, conf=conf["DETECTION"])

# encoder = encoding.FacialEncoder(conf)
# encoder = encoding_arc.EncodingArc(conf)

# matcher = default.FaceMatcher(conf)
# matcher = svm.SVMMatcher(conf)
# matcher = knn.KNNMatcher(conf)


# In[6]:

for _ in range(LOOPS):
    _ = detector.extract_datum(image)


# In[5]:


del faces


# In[ ]:


# if len(faces) > 0:
#    face_images = [face.face_image for face in faces]

#    for idx, emb in enumerate(encoder.predict(face_images)):
#        face = faces[idx]  # type: datum.Datum
#        face.embedding = emb.reshape((1, -1))
#        face.identity, face.match_score = matcher.recognize(face.embedding)


# In[ ]:


# conf
