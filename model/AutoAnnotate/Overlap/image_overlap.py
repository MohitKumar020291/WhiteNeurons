# Not Quite impressed with the code, works for now

import cv2

from pipeline.load_ss import load_ss
from pipeline.loadAnnotedData.helper import show_image



def overlap_function() -> None:
    images = load_ss(off=2)
    # gray scaling images
    images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    images[1] = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    diff = images[0] - images[1]
    return diff


def main() -> None:
    diff = overlap_function()
    # show_image(diff)
    return None



if __name__ == "__main__":
    main()