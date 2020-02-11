#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
# get_ipython().magic(u'matplotlib inline')

import cv2

from face_recognition_ros import detection, encoding_arc
from face_recognition_ros.utils import config

# os.environ['MXNET_CPU_WORKER_NTHREADS'] = '4'
# os.environ['MXNET_CPU_PRIORITY_NTHREADS'] = '8'
# os.environ['OMP_NUM_THREADS'] = '8'
# os.environ['MXNET_CPU_NNPACK_NTHREADS'] = '8'
# os.environ['MXNET_MP_OPENCV_NUM_THREADS'] = '1'

# sys.path.append("/home/sam/catkin_ws/src/src/face_recognition_ros/src")


# In[2]:
# TENSOR: 66 (+preprocessing)
# MXNET: 530 (+preprocessing)


# IMAGE_PATH = "/home/sam/Pictures/IMG-20190419-WA0001.jpg"
IMAGE_PATH = "/home/sam/UMA/4/4_2/3-TFG/3-Workspace/face_recognition_ros/face_recognition_ros/data/database/family_dataset/sam/IMG_20190730_182021.jpg"
image = cv2.cvtColor(cv2.imread(IMAGE_PATH, 1), cv2.COLOR_BGR2RGB)

# RESOLUTION = (1280, 720) # 0.433
RESOLUTION = (640, 480)  # 0.213
METHOD = "mtcnn"

LOOPS = 50

image = cv2.resize(image, RESOLUTION)
# plt.imshow(image)


# In[3]:


conf = config.load_config()

detector = detection.FacialDetector(method=METHOD, conf=conf["DETECTION"])

# encoder = encoding.FacialEncoder(conf)
encoder = encoding_arc.EncodingArc(conf)

# matcher = default.FaceMatcher(conf)
# matcher = svm.SVMMatcher(conf)
# matcher = knn.KNNMatcher(conf)


# In[6]:

# for _ in range(LOOPS):
faces = detector.extract_datum(image)

# In[ ]:

for _ in range(LOOPS):

    if len(faces) > 0:
        face_images = [face.face_image for face in faces]

        for idx, emb in enumerate(encoder.predict(face_images)):
            pass
            # face  = faces[idx]  # type: datum.Datum
            # face.embedding = emb.reshape((1, -1))
            # face.identity, face.match_score = matcher.recognize(face.embedding)


# In[ ]:


# conf
