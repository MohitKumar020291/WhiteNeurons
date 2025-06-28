import cv2
from numpy import ndarray as npnda
from typing import Union
from skimage.color import label2rgb

# modules are running individually
# python3 -m model.AutoAnnotate.tests.segment_test
from pipeline.loadAnnotedData import show_image
from utils import image2array


def visual_segments(type_: str, segments: Union[npnda, str], image, show=True):
    """
    Args:
        type_:
            - org: shows the real b/w image - normalized version
            - color: color codes the org image
            - overlay: overlays over the real image
        image: either real image array or path to read image from
    """
    types = ['org', 'color', 'overlay']
    assert type_ in types, f"type_ should belong to {types}"
    if type_ == 'overlay' and image is None:
        raise Exception(f"The original image, image is required, provided {image}")
    image = image2array(image)

    segments_normalized = (segments * 255 / segments.max()).astype('uint8')
    # segments_colored = cv2.applyColorMap(segments_normalized, cv2.COLORMAP_JET)
    segments_colored = label2rgb(segments, image=image, bg_label=0) # maps each label (region id) to a different colors
    segments_overlayed = cv2.addWeighted(
        image, 0.7,
        cv2.applyColorMap(
            (segments * 255 / segments.max()).astype('uint8'),
            cv2.COLORMAP_JET
        ), 0.3, 0
    )

    if show:
        if type_ == 'org':
            show_image(segments_normalized)
        elif type_ == 'color':
            show_image(segments_colored)
        else:
            show_image(segments_overlayed)
    else:
        images = [segments_normalized, segments_colored, segments_overlayed]
        return images[types.index(type_)]
    
def visual_image(image) -> None:
    image = image2array(image) if not isinstance(image, npnda) else image
    show_image(image)
    return None


