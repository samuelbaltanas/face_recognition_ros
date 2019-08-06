class BaseFaceDetector:
    def __init__(self):
        raise NotImplementedError()

    def extract_region(self, image):
        raise NotImplementedError()

    def extract_images(self, image, regions=None, raw_detection=None):
        if regions is None:
            regions, _ = self.extract_region(image)
        return [r.extract_face(image) for r in regions]
