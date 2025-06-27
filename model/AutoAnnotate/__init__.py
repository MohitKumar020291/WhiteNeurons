from .Segmentself.segment import felzenszwalb
# from .Segmentself import felzenszwalb
from .helper import visual_segments, visual_image, merge_similar_regions
from .Segmentsk.segment import image_segmenter

__all__ = [
    'felzenszwalb',
    'visual_segments',
    'visual_segments',
    'image_segmenter',
    'merge_similar_regions'
]