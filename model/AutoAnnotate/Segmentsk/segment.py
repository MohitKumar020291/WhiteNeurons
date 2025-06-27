from numpy import ndarray as npnda
from skimage.segmentation import felzenszwalb

from typing import Union
from utils import image2array

def image_segmenter(image: Union[str, npnda], scale: float = 100, sigma: float = 0.8):
    image = image2array(image)
    shape = list(image.shape)
    try:
        channel_axis = shape.index(3) 
    except:
        channel_axis = None 

    args = {
        "image": image,
        "scale": scale,
        "sigma": sigma,
        "channel_axis": channel_axis
    }

    # each pixel of this image is a region id
    segmened_image = felzenszwalb(**args)
    return segmened_image