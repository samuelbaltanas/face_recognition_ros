from face_recognition_ros import datum, detection, encoding, encoding_arc
from face_recognition_ros.classifiers import default, knn, svm


class Recognition:
    def __init__(self, config=None):
        self.detector = detection.FacialDetector(config)

        # self.encoder = encoding.FacialEncoder(config)
        self.encoder = encoding_arc.EncodingArc(config)

        self.matcher = default.FaceMatcher(config)
        # self.matcher = svm.SVMMatcher(config)
        # self.matcher = knn.KNNMatcher(config)

    def recognize(self, image):
        faces = self.detector.predict(image, extract_image=True)
        if len(faces) > 0:
            face_images = [face.image for face in faces]

            for idx, emb in enumerate(self.encoder.predict(face_images)):
                face = faces[idx]  # type: datum.Datum
                face.embedding = emb.reshape((1, -1))
                face.identity, face.match_score = self.matcher.recognize(
                    face.embedding
                )

        return faces
