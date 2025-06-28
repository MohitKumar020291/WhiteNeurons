from PIL import Image
import numpy as np


from ....AutoAnnotate import merge_similar_regions, image_segmenter, visual_segments


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


if __name__ == "__main__":
    test_merge_segments(image='path', type_='color')