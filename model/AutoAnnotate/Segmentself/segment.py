import numpy as np
# from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from skimage import img_as_float64
from numba.experimental import jitclass
from numba import njit, int32, float64

disJointSetSpec = {
    'parent': int32[:],
    'rank': int32[:],
    'size': int32[:],
    'int_diff': float64[:]
}

# @jitclass(disJointSetSpec)
class DisjointSet:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)
        self.int_diff = np.zeros(size, dtype=np.float64)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y, weight, k):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return
        
        tau_x = k / self.size[x_root]
        tau_y = k / self.size[y_root]
        m_int = min(self.int_diff[x_root] + tau_x, 
                   self.int_diff[y_root] + tau_y)
        
        if weight > m_int:
            return
        
        if self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
            self.size[x_root] += self.size[y_root]
            self.int_diff[x_root] = max(weight, self.int_diff[x_root], self.int_diff[y_root])
        else:
            self.parent[x_root] = y_root
            self.size[y_root] += self.size[x_root]
            self.int_diff[y_root] = max(weight, self.int_diff[x_root], self.int_diff[y_root])
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[y_root] += 1

# Compute edge weights (8-connectivity)
def compute_edges(blurred, num_pixels):
    height, width = blurred.shape[:2]
    idx_map = np.arange(num_pixels).reshape(height, width)

    diff_right = np.sqrt(np.sum((blurred[:, 1:] - blurred[:, :-1]) ** 2, axis=-1)).ravel()
    edges_right = np.c_[
        idx_map[:, :-1].ravel(),
        idx_map[:, 1:].ravel()
    ]

    diff_down = np.sqrt(np.sum((blurred[1:] - blurred[:-1]) ** 2, axis=-1)).ravel()
    edges_down = np.c_[
        idx_map[:-1].ravel(),
        idx_map[1:].ravel()
    ]

    diff_dright = np.sqrt(np.sum((blurred[1:, 1:] - blurred[:-1, :-1]) ** 2, axis=-1)).ravel()
    edges_dright = np.c_[
        idx_map[1:, 1:].ravel(),
        idx_map[:-1, :-1].ravel()
    ]

    diff_dleft = np.sqrt(np.sum((blurred[:-1, 1:] - blurred[1:, :-1]) ** 2, axis=-1)).ravel()
    edges_dleft = np.c_[
        idx_map[1:, :-1].ravel(),
        idx_map[:-1, 1:].ravel()
    ]

    edges = np.vstack([edges_right, edges_down, edges_dright, edges_dleft])
    weights = np.hstack([diff_right, diff_down, diff_dright, diff_dleft])
    return edges, weights
    

def get_sorted_edge_array(edges, weights):
    # list[tuple[(name, dtype)]]
    dtype = np.dtype([('u', np.int32), ('v', np.int32), ('w', np.float64)])
    edge_array = np.zeros(edges.shape[0], dtype=dtype)
    edge_array['u'] = edges[:, 0]
    edge_array['v'] = edges[:, 1]
    edge_array['w'] = weights
    edge_array.sort(order='w')
    return edge_array

def felzenszwalb(image, scale=100, sigma=0.8, min_size=20):
    """Python implementation of Felzenszwalb's efficient graph based segmentation.
    
    Args:
        image: Input image (H, W, C)
        scale: Scale parameter controlling segment size
        sigma: Width of Gaussian smoothing kernel
        min_size: Minimum component size
    
    Returns:
        Integer mask indicating segment labels (H, W)
    """
    # Convert and preprocess image
    image = img_as_float64(image)
    scale = float(scale) / 255.
    
    # Apply Gaussian blur (channel-wise)
    blurred = np.zeros_like(image)
    for c in range(image.shape[2]):
        # blurred[..., c] = gaussian_filter(image[..., c], sigma=sigma)
        blurred[..., c] = gaussian(image[..., c], sigma=sigma)
    
    height, width = image.shape[:2]    
    num_pixels = height * width


    # Get all edges and weights
    edges, weights = compute_edges(blurred, num_pixels)

    edge_array = get_sorted_edge_array(edges, weights)
    
    # Initialize Union-Find
    uf = DisjointSet(num_pixels)
    
    # First pass: merge with condition
    for u, v, w in edge_array:
        uf.union(u, v, w, scale)

    for u, v, _ in edge_array:
        root_u = uf.find(u)
        root_v = uf.find(v)
        if root_u != root_v and (uf.size[root_u] < min_size or uf.size[root_v] < min_size):
            uf.union(u, v, 0, scale)
    
    # Generate output labels
    labels = np.arange(num_pixels)
    for i in range(num_pixels):
        labels[i] = uf.find(i)
    
    # Remap labels to consecutive integers
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.vectorize(label_map.get)(labels.reshape(height, width))
    
    return final_labels

# 4.331