from cvgeomkit.common import ArrayLike, NumpyImage


class CourtDetector:

    def __init__(self, img: ArrayLike):
        self.img = NumpyImage(img)
        

        self.center_crop_img = 