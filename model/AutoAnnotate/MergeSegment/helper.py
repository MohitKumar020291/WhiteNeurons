from skimage.measure import regionprops
from skimage import graph
from numpy.linalg import norm


def uint8_conversion(image):
    return (image * 255).astype('uint8')

def calculate_size_diff(size_src, size_dst):
    size_diff = abs(size_src - size_dst) / max(size_src, size_dst)
    return size_diff

def get_minfo(rag, src, dst):
    src_node = rag.nodes[src]
    dst_node = rag.nodes[dst]

    size_src = src_node['pixel count']
    size_dst = dst_node['pixel count']

    total_count = size_src + size_dst
    new_mean = (src_node['mean color'] * size_src + dst_node['mean color'] * size_dst) / total_count

    return size_src, size_dst, total_count, new_mean

# Source: https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_rag_merge.html
def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = norm(diff)
    return {'weight': diff}


def merge_criteria_color(rag, src, dst):
    """
    This function defines the logic of merging two regions: src and dst
    """

    # mean color difference between two regions in lab form
    # color_diff = src_nodes['mean color'] - dst_nodes['mean color']
    # # certainity: what if the difference is negative but high? that's why we have that
    # color_dist = norm(color_diff)

    # normalized size difference between the two regions
    # * 100: gives percentage of change i.e. 10, 8
    # gives: 0.2
    
    size_src, size_dst, total_count, new_mean = get_minfo(rag, src, dst)

    size_diff = calculate_size_diff(size_src, size_dst)

    # return color_dist < 20 and size_diff < 0.5
    if size_diff < 0.5:
        rag.nodes[dst]['mean color'] = new_mean
        rag.nodes[dst]['pixel count'] = total_count
        return True


def merge_color_basis(rag, segments):
    """
    returns:
        merged segments on the basis of the color similarity between real image
    """
    # role of merging is like:
    # if weight_func(rag, src, dst, n)["weight"] < thresh:
    #     merge_func(...)
    merged_segments = graph.merge_hierarchical(
                            segments, rag, 
                            thresh=30, #takes care out of the color difference
                            rag_copy=False,
                            in_place_merge=True,
                            merge_func=merge_criteria_color,
                            weight_func=_weight_mean_color
                    )
    return merged_segments


def merge_criteria_texture(rag_gray, src, dst):
    size_src, size_dst, total_count, new_mean = get_minfo(rag_gray, src, dst)

    size_diff = calculate_size_diff(size_src, size_dst)

    # return color_dist < 20 and size_diff < 0.5
    if size_diff < 0.5:
        rag_gray.nodes[dst]['mean color'] = new_mean
        rag_gray.nodes[dst]['pixel count'] = total_count
        return True


def merge_texture_basis(rag_gray, segments):
    # storing the texture per node
    merged_segments = graph.merge_hierarchical(
                        segments, rag_gray,
                        thresh=0.03,
                        rag_copy=False,
                        in_place_merge=True,
                        merge_func=merge_criteria_texture,
                        weight_func=_weight_mean_color
                    )
    return merged_segments


def is_shape_merge_reasonable(prop_src, prop_dst, src_mask, dst_mask, solidity_thresh=0.9, ecc_thresh=0.95):
    from numpy import logical_or
    from skimage.measure import regionprops

    combined_mask = logical_or(src_mask, dst_mask).astype('uint8')
    combined_props = regionprops(combined_mask)

    if not combined_props:
        return False

    combined_prop = combined_props[0]
    return combined_prop.solidity > solidity_thresh and combined_prop.eccentricity < ecc_thresh


def is_noise(prop_src, prop_dst):
    area_ratio = prop_src.area / prop_dst.area
    if not (0.3 < area_ratio < 0.7):
        return True
    if prop_src.solidity < 0.3 or prop_dst.solidity < 0.3:
        return True
    return False


def merge_criteria_shape(rag, src, dst, segments):
    """
        IOU would not work directly
    """
    props = get_regionprops_prop(segments, src, dst, mask=True)
    prop_src = props[src]
    prop_dst = props[dst]
    mask_src = (segments == src)
    mask_dst = (segments == dst)

    to_merge = is_shape_merge_reasonable(prop_src, prop_dst, mask_src, mask_dst)
    noise = is_noise(prop_src, prop_dst) # if src is a small noise region

    if to_merge and not noise:
        _, _, total_count, new_mean = get_minfo(rag, src, dst)
        rag.nodes[dst]['mean color'] = new_mean
        rag.nodes[dst]['pixel count'] = total_count
        return True
    return False


def merge_shape_basis(rag, segments):
    merged_segments = graph.merge_hierarchical(
                        segments, rag,
                        thresh=0.05,
                        rag_copy=False,
                        in_place_merge=True,
                        merge_func=merge_criteria_shape,
                        weight_func=_weight_mean_color
                    )
    return merged_segments


def get_metadata_props(props, labels=None):
    """
    returns:
        meta data about props 
    """
    import math

    props_dict = {}

    for i, prop in enumerate(props):
        y0, x0 = prop.centroid
        orientation = prop.orientation
        # vertex of boundaries of a rectangle
        x1 = x0 + math.cos(orientation) * 0.5 * prop.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * prop.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * prop.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * prop.axis_major_length

        minr, minc, maxr, maxc = prop.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        attrs = {
            'boundary': (x1, y1, x2, y2),
            'box': (bx, by),
            'prop': prop
        }
        # label = prop.label # how to get this label
        label = labels[i] if labels else prop.label
        props_dict[label] = attrs
    
    return props_dict

def centroid_distance(props, src, dst):
    from numpy import array
    # I have already imported outside
    dist = norm(array(props[src]['centroid']) - array(props[dst]['centroid']))
    return dist

def get_regionprops_prop(segments, mask: bool=None, src: int=None, dst: int=None, all_: bool=False):
    """
    The current function considers a mask region - i.e. masked_segment = segments == label
    and the props for all the regions could be found directly if the all arg is True
    regions = regionprops(segments)

    ------------------------------
    returns:
        a well formed dictionary of the regions or a single segment
    """
    import numpy as np
    assert mask or all_, f"Provide either mask or all_ provided mask={mask} and all_={all_}"
    if mask:
        if src is None or dst is None:
            raise Exception(f"for mask provide src and dst both, provide {src}, {dst}")
        if all_:
            raise Exception(f"either get props on the basis of the mask={mask} or all_={all_}")

        # source: https://scikit-image.org/docs/0.23.x/auto_examples/segmentation/plot_regionprops.html
        mask_src = (segments == src)
        mask_dst = (segments == dst)

        if not np.any(mask_src) or not np.any(mask_dst):
            return None

        props_src = regionprops(mask_src.astype('int'))[0]
        props_dst = regionprops(mask_dst.astype('int'))[0]

        # for each prop there will be two labels - 0 and 1
        # but 0 is considered as background
        props = [props_src, props_dst]

        props_dict = get_metadata_props(props, labels = [src, dst])

        return props_dict
    
    elif all_:
        props = regionprops(segments)

        props_dict = get_metadata_props(props)

        return props_dict
    
def get_small_segments(segments, threshold):
    """
    returns:
        bool mask where non zero are the regions_id having the number of pixels < threshold
    """
    import numpy as np
    from skimage.measure import regionprops

    props = regionprops(segments)
    max_label = segments.max() + 1
    pixel_counts = np.zeros(max_label, dtype=int)
    
    for prop in props:
        pixel_counts[prop.label] = prop.area

    small_segments_mask = pixel_counts < threshold
    return small_segments_mask


def constant_weight(graph, src, dst, n):
    return {'weight': 1}


class MergeStateSize:
    def __init__(self, small_segments_dict, type_, rag):
        self.small_segments_dict = small_segments_dict
        self.type_ = type_
        if self.type_ == 'color':
            self.merge_criteria_func = merge_criteria_color

    def merge_func(self, rag, src, dst):
        if self.small_segments_dict.get(src, False) or self.small_segments_dict.get(dst, False):
            print(self.small_segments_dict[src], self.small_segments_dict[dst])
            merged = self.merge_size_basis(rag, src, dst)
            if merged:
                self.small_segments_dict[src] = False
                self.small_segments_dict[dst] = False
            return merged
        return False
    
    def merge_size_basis(self, rag, src, dst):
        return self.merge_criteria_func(rag, src, dst)