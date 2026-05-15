from cvgeomkit.common import ArrayLike

from src.utils.validators import check_if_numpy_image, validate_number


def crop_center_img(
        img: ArrayLike, 
        crop_ratio: float = 0.4
) -> tuple[ArrayLike, int, int]:
        
        validate_number(crop_ratio, float, 0, 1, min_inclusive=False)
        
        img = check_if_numpy_image(img)
        
        w = img.width

        margin = int((1 - crop_ratio) * w / 2)
        crop = img[:, margin:w - margin]

        ch, cw = crop.height, crop.width

        return crop, ch, cw
