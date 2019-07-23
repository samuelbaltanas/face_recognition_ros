from face_recognition_ros import (
    encoding,
    detection,
    matching,
    datum,
    image_preprocessing,
)


class Recognition:
    def __init__(self, config=None):
        self.detector = detection.FacialDetector(config)
        self.encoder = encoding.FacialEncoder(config)
        self.matcher = matching.FaceMatcher(config)

    def recognize(self, image):
        # DONE Preprocessing

        faces = self.detector.extract(image, 0.1)
        face_images = [
            image_preprocessing.preprocess_face(face.face_image) for face in faces
        ]

        for idx, emb in enumerate(self.encoder.predict(face_images)):
            face = faces[idx]  # type: datum.Datum
            face.face_image = face_images[idx]
            face.embedding = emb
            (face.identity, _), face.match_score = self.matcher.recognize(
                face.embedding
            )

        return faces
