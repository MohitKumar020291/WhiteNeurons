from utils import refine_mask_with_value


def refine_mask(mask, hsvImage, v_threshold=80):
    """returns refine mask with value from HSV"""
    return refine_mask_with_value(mask, hsvImage, v_threshold)