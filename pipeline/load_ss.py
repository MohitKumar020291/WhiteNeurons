from model.AutoAnnotate.UNet.dataset_loading import load_train_data
from pipeline.loadAnnotedData.helper import show_image
from utils import image2array

import torch
import cv2
import numpy as np
import os
from PIL import Image



def directory_drama():
    ss = "ss"
    current_directory = os.path.dirname(__file__)
    white_neurons_dir = os.path.dirname(current_directory)
    ss_directory = os.path.join(white_neurons_dir, ss)
    return ss_directory


def get_images(ss_directory: str) -> torch.tensor:
    images = list()
    images_name = os.listdir(ss_directory)
    for image_name in images_name:
        image_path = os.path.join(ss_directory, image_name)
        image = Image.open(image_path)
        image = image2array(image)
        images.append(image)
    return images


# Use less for now
def get_w_h():
    train_dataset = load_train_data(type_="valid")
    seg_image = train_dataset[0]
    _, w, h = seg_image.org_image.shape # c, w, h, they are w, h not h, w
    return w, h


def resize_image(image, w, h, show=False):
    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    # for validation
    if show:
        show_image(image=resized_image)
    return resized_image


def load_ss() -> list[np.ndarray]:
    "this is main of this file :)"
    ss_directory = directory_drama()
    w, h = 640, 640
    images = get_images(ss_directory)
    for idx, image in enumerate(images):
        resized_image = resize_image(image, w, h, show=True)
        images[idx] = resized_image
    return images


if __name__ == "__main__":
    load_ss()
    