from utils import (
                get_binary_mask,
                get_hsv_mask,
                get_hsv_from_masked_region
                )
import numpy as np



def apply_mask(
        frame: np.array,
        use_binary_mask: bool,
        show_by: bool,
        color: list[int]
        ):
    """Applies mask to the frame"""
    if use_binary_mask:
        mask = get_binary_mask(frame)
        hsvImage = get_hsv_from_masked_region(frame, mask)
        # hsvImage = None
    else:
        mask, hsvImage = get_hsv_mask(frame, color)

    return mask, hsvImage