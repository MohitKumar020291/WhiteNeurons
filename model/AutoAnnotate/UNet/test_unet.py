import torch
import numpy as np
import os

from .dataset_loading import load_infer_data
from model.AutoAnnotate.helper import return_org_image_and_label_only, visual_segments
from model.AutoAnnotate.UNet.function_blocks import UNet
from model.AutoAnnotate.helper import test_init, load_models, post_process_infer
from model.AutoAnnotate.helper import show_image




def main():
    # should read the cl-arg for the visual_segments
    torch.set_default_device("cuda")

    dataset = load_infer_data(type_="test")
    cats = dataset[0].cats

    current_directory = os.path.dirname(__file__)
    train_config, checkpoint_dir = test_init(current_directory)
    models = load_models(train_config, checkpoint_dir, cats)

    for cat_id, cat in enumerate(cats):
        print(cat)
        model = UNet()
        model.load_state_dict(models[cat]['model_state_dict'])

        for i in range(len(dataset)):
            data = dataset[i]
            image, _ = return_org_image_and_label_only(data, cat_id)
            output_segs = model(image.permute(0, 3, 1, 2).float()) # Handle through a function
            image_numpy, output_segs_numpy_normalized = post_process_infer(image, output_segs)
            visual_segments(type_='overlay', segments=output_segs_numpy_normalized, image=image_numpy, window_name=f"CATEGORY = {cat}")


if __name__ == '__main__':
    main()
    # import cv2
    # dirc = "/home/mohitb1i/pytorch_env/WhiteNeurons/ss"
    # images = os.listdir(dirc)
    # image = cv2.imread(os.path.join(dirc, images[0]))
    # print(image.shape)
    # from utils import image2array
    # image = image2array(image)
    # print(image.shape)