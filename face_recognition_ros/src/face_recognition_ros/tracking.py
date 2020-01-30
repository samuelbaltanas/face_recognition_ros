import logging

import cv2

from face_recognition_ros import core


logger = logging.getLogger(__name__)


class Track:
    def __init__(self, tracker, ttd):
        self.tracker = tracker
        self.ttd = ttd


class MultiTracker:
    def __init__(self):
        self.TTD = 4
        self.tracklets = []

    def init(self, image, region: core.region.RectangleRegion):

        tracklet = cv2.TrackerCSRT_create()
        # tracklet = cv2.TrackerKCF_create()

        retval = tracklet.init(image, region.to_cvbox(margin=0.0))

        if retval:
            self.tracklets.append(Track(tracklet, self.TTD))
            logger.debug("Tracklet initialization success")
        else:
            logger.debug("Tracklet initialization faillure")

    def update(self, image):
        res = []
        rem_list = []
        for i, tr in enumerate(self.tracklets):
            retval, bbox = tr.tracker.update(image)
            if retval:
                reg = core.region.RectangleRegion(bbox[0], bbox[1], bbox[2], bbox[3])
                res.append(reg)
            else:
                # DONE Destroy tracker condition
                logger.debug("Tracklet face retrieval faillure.")
                tr.ttd -= 1
                if tr.ttd == 0:
                    rem_list.append(i)

        # Remove all tracklets where ttd == 0
        for i in rem_list:
            self.tracklets.pop(i)

        return res

    def is_tracking(self):
        return len(self.tracklets) > 0

    def add(self, image, region: core.region.RectangleRegion):
        # TODO Check if bbox is being tracked already.
        pass
