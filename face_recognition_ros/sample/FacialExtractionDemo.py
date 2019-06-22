#!/usr/bin/env python
# coding: utf-8

# In[4]:


#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')
#get_ipython().magic(u'matplotlib inline')
import sys
import os

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

# CORE PROJECT
sys.path.append('/home/sam/catkin_ws/src/src/face_recognition_ros/src')
# from face_recognition_ros.utils import files

# FACENET
# sys.path.append(os.path.join(files.PROJECT_ROOT, '../facenet/src'))

# OPENPOSE
# sys.path.append('/usr/local/python')
# from openpose import pyopenpose as op

from face_recognition_ros import detection, encoding, matching, storage
from face_recognition_ros.utils import config, files
from face_recognition_ros.extraction import oriented_bounding_box, aligned_bounding_box


# GLOBALS 
conf = config.load_config()["OPENPOSE"]
det = detection.FacialDetector(conf)


# # Facial recognition in ROS
# 
# ## Face extraction
# 
# For the facial detection and extraction we will use OpenPose (...).
# 
# ### Default algorithm (MTCNN)
# 
# The default face detector used in Facenet is an implementation of Multi-task CNN (mtcnn). Therefore we need to compare its performance to ours.

# In[5]:


PERSON = 6

# IMG_PATH = files.get_flw_sample_path(PERSON)[0]
# IMAGE_PATH = IMG_PATH.replace("flw_mtcnnpy_160", "raw", 1).replace( ".png", ".jpg", 1)

IMAGE_PATH = '/home/sam/Pictures/IMG-20190419-WA0001.jpg'

image = cv.imread(IMAGE_PATH, 1)

def mtcnn_demo(image_path):  
    image1 = encoding.load_images([image_path])
    #image2 = face_ros.encoding.load_images([IMG_PATH])

    print "Faces detected: {}".format(image1.shape)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax2.imshow(image1[0])
    #ax3.imshow(image2[0])
    
#mtcnn_demo(IMAGE_PATH)


# ### Our approach (OpenPose)
# 

# In[6]:


def openpose_demo(image):  

    datum = det.extract_keypoints(image)

    #f, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.imshow(cv.cvtColor(datum.cvOutputData, cv.COLOR_BGR2RGB))
    
    return datum


# In[7]:


# %%time
datum = openpose_demo(image)


# In[18]:


THRES = 0.25

#bbs = oriented_bounding_box.extract_from_pose(datum, confidence_threshold=THRES)
bbs = aligned_bounding_box.extract_from_pose(datum, confidence_threshold=THRES)
print 'Faces detected {}'.format(len(bbs))
for i in bbs:
    print i.shape
# utils.images.plot_bounding_box(image, bbs, color_enc=cv.COLOR_BGR2RGB)


# In[13]:


I = 0

aligned_bounding_box.plot_bounding_box(image, bbs, cv.COLOR_BGR2RGB)

#print bbs
plt.figure()
img = aligned_bounding_box.extract_image_faces(bbs, image)[I]
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


# In[14]:


bbs


# In[ ]:


#with tf.Graph().as_default():
    #with tf.Session() as sess:
        #face_enc = encoding.FacialEncoder(sess,
        #                                  config.CONFIG["FACENET"]["model"])
        #image = cv.imread(files.get_flw_sample_path(0)[0], 1)
        #emb = face_enc.predict([fac])
        #print emb


# In[ ]:




