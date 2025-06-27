# File contains code for the benchmarking between my implementation of segmentation and sklearn's

from .. import visual_segments
from PIL import Image
from utils import bench

def bench_segment():
    # image_path = "../../Downloads/65534image1.jpg"
    # image = Image.open(image_path)
    from skimage import data, io
    image = data.coffee()
    sum = 0
    for i in range(10):
        sum += bench(visual_segments, {"type_": 'org', "image": image, "show": False})
    print(sum / 10)

    # from skimage.segmentation import felzenszwalb
    # time_sk = bench(felzenszwalb, {"image": image,  "scale": 100, "sigma": 0.8, "min_size": 20})
    # print(time_self - time_sk)

if __name__ == "__main__":
    bench_segment()