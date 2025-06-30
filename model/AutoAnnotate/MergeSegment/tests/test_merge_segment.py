from PIL import Image
import numpy as np

try:
    from model.AutoAnnotate import merge_similar_regions
except:
    raise Exception("Cannot import merge_similar_regions")
try:
    from model.AutoAnnotate import image_segmenter
except:
    raise Exception("Cannot import merge_similar_regions")
try:
    from model.AutoAnnotate import visual_segments 
except:
    raise Exception("Cannot import merge_similar_regions")
try:
    from model.AutoAnnotate import merge_smaller_segments
except:
    raise Exception("Cannot import merge_similar_regions")

from utils import image2array


def test_merge_segments(image: str, type_: str) -> None:
    if image == 'sk':
        from skimage import data, io
        image = data.coffee()
    else:
        image_path = "../../Downloads/65534image1.jpg"
        image = Image.open(image_path)

    segments = image_segmenter(image)

    # types = ['color', 'texture', 'shape'] # I don't know if this is correct order or not
    types = ['texture', 'color'] # I don't know if this is correct order or not
    # types = ['color'] # I don't know if this is correct order or not
    merged_segments = segments
    for _ in range(2):
        for type_ in types:
            merged_segments = merge_similar_regions(type_=type_, image=image, segments=segments)

    segments_before = len(np.unique(segments))
    segments_after = len(np.unique(merged_segments))
    print(f"# Segments reduced {segments_before - segments_after}")
    visual_segments(type_='color', segments=merged_segments, image=image)


def test_merge_segments_shape():
    from ....AutoAnnotate import image_segmenter
    from skimage import data
    import numpy as np

    # image = data.coffee()
    image_path = "ss/mask_ss_17.png"
    image = Image.open(image_path)
    image = image2array(image)
    segments = image_segmenter(image)
    print("Before merging", len(np.unique(segments)))
    small_merged_segments = merge_smaller_segments(image, segments, threshold=0.05) 
    print("After merging", len(np.unique(small_merged_segments)))
    visual_segments(type_='color', segments=small_merged_segments, image=image)

    # extra segments after the 
    type__based_merged_segs = small_merged_segments # Initial segments
    for type_ in ['color', 'shape', 'texture']:
        # each time a new segment not accumulating but using the initial segment as small_merged_segments
        merged_segs = merge_similar_regions(type_=type_, image=image, segments=small_merged_segments)
        visual_segments(type_='overlay', segments=merged_segs, image=image)

if __name__ == "__main__":
    # test_merge_segments(image='path', type_='color')
    test_merge_segments_shape()