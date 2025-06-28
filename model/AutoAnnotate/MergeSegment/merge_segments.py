from ...AutoAnnotate import (
                uint8_conversion, 
                merge_color_basis, 
                merge_texture_basis, 
                merge_shape_basis,
                get_small_segments,
                visual_segments,
                MergeStateSize,
                constant_weight,
                _weight_mean_color
            )
from utils import image2array

from skimage.color import rgb2lab, rgb2gray
from skimage import graph

def merge_similar_regions(type_: str, image, segments):
    """
    image: original image
    segments: segmented image
    """
    types = ['color', 'texture', 'shape']
    assert type_ in types, f"the type_ should be one of {types}"
    image = image2array(image)

    if type_ in ['color', 'shape']:
        # Nodes = image regions
        # Edges = connections between adjacent regions
        # the distance is based on the similarity
        # adjency graph with
        # keys: node/region and values: weight with all other node/regions
        if type == 'color':
            image = rgb2lab(image)
        rag = graph.rag_mean_color(image, segments, mode='similarity')

    if type_ == 'color':
        # launch the function for the color similarity
        return merge_color_basis(rag, segments)

    if type_ == 'texture':
        gray_scale_image = rgb2gray(image)
        # mean color of rag_gray is just the intensity repeated three times
        # i.e. rag_gray.nodes[500]['mean color'] = [0.39357495 0.39357495 0.39357495]
        rag_gray = graph.rag_mean_color(gray_scale_image, segments, mode='similarity')

        return uint8_conversion(merge_texture_basis(rag_gray, segments))

    if type_ == 'shape':
        return merge_shape_basis(rag, segments)


def merge_smaller_segments(image, segments, threshold):
    rag = graph.rag_mean_color(image, segments)
    
    import numpy as np
    small_segments = get_small_segments(segments, threshold)
    print("Small segments true", np.count_nonzero(small_segments))

    small_segments_dict = {
        region_id: True
        for region_id in small_segments
        if region_id
    }
    state = MergeStateSize(small_segments_dict, type_='color', rag=rag)

    merged_segments = graph.merge_hierarchical(
        segments,
        rag,
        thresh=20,
        rag_copy=False,
        in_place_merge=True,
        merge_func=state.merge_func,
        weight_func=_weight_mean_color,
    )

    return merged_segments


def ongoing_test():
    from ...AutoAnnotate import image_segmenter
    from skimage import data
    import numpy as np

    image = data.coffee()
    segments = image_segmenter(image)
    print("Before merging", len(np.unique(segments)))
    small_merged_segments = merge_smaller_segments(image, segments, threshold=0.05)
    print("After merging", len(np.unique(small_merged_segments)))
    visual_segments(type_='color', segments=small_merged_segments, image=image)




if __name__ == "__main__":
    ongoing_test()
    