from ....AutoAnnotate import visual_segments
from ....AutoAnnotate import image_segmenter

def test_visual():
    # image_path = "../../Downloads/65534image1.jpg"
    # image = Image.open(image_path)
    from skimage import data, io
    image = data.coffee()
    segments = image_segmenter(image)
    visual_segments(segments=segments, type_='overlay', image=image)


if __name__ == "__main__":
    test_visual()