class Datum:
    def __init__(
        self,
        pose=None,
        pose_score=0.0,
        face_region=None,
        face_image=None,
        embedding=None,
        identity=None,
        match_score=4.0,
    ):
        # POSE
        self.pose = pose
        self.pose_score = pose_score
        # DETECTION
        self.face_region = face_region
        self.face_image = face_image
        # RECOGNITION
        self.embedding = embedding
        # MATCHING
        self.identity = identity
        self.match_score = match_score

    def __str__(self):
        return "\n".join(
            [
                # "Pose: {}".format(self.pose),
                "Pose score: {}".format(self.pose_score),
                "Region: {}".format(self.face_region),
                # "Image: {}".format(self.face_image),
                # "Embedding: {}".format(self.embedding),
                "Identity: {}".format(self.identity),
                "Match score: {}".format(self.match_score),
            ]
        )
