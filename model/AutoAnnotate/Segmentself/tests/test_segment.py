from PIL import Image

from ....AutoAnnotate import visual_segments
from ....AutoAnnotate import felzenszwalb

def test_visual():
    # image_path = "../../Downloads/65534image1.jpg"
    # image = Image.open(image_path)
    from skimage import data, io
    image = data.coffee()
    segments = felzenszwalb(image)
    visual_segments(segments=segments, type_='overlay', image=image)

if __name__ == "__main__":
    test_visual()