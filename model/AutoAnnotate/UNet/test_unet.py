import torch
import numpy as np
import os
from typing import Tuple, Dict, AnyStr

from utils import read_yaml_file
from .dataset_loading import load_infer_data
from model.AutoAnnotate.helper import return_org_image_and_label_only, show_image, visual_segments
from model.AutoAnnotate.UNet.function_blocks import UNet


def infer_init() -> Tuple[Dict, AnyStr]:
    current_directory = os.path.dirname(__file__)
    train_config_path = os.path.join(current_directory, 'unet_config.yaml')
    train_config = read_yaml_file(train_config_path)

    checkpoint_dir = train_config.get("checkpoint_dir")
    if checkpoint_dir is None:
        print(f"provide the checkpoints dir, where models are stored!, provided = {checkpoint_dir}")
    checkpoint_dir = os.path.join(current_directory, checkpoint_dir)
    
    return train_config, checkpoint_dir


def load_models(train_config, checkpoint_dir: str, cats: list[str]) -> Dict:
    epochs = train_config.get("epochs", None)
    assert epochs != None, "There are no epochs in train_config"
    models = dict()
    files = os.listdir(checkpoint_dir)
    for file in files:
        if '16' in file:
            model_path = checkpoint_dir + f"/{file}"
            try:
                cat = cats[int(file.split("_")[1][-1])]
                models[cat] = torch.load(model_path)
            except FileNotFoundError:
                print(f"Error: File not found at {model_path}")
                exit()
            except Exception as e:
                print(f"An error occurred while loading the state dictionary: {e} from path = {model_path}")
                exit()
    return models


def visualize_overlayed_image():
    ...


def main():
    torch.set_default_device("cuda")

    dataset = load_infer_data(type_="test")
    cats = dataset[0].cats

    train_config, checkpoint_dir = infer_init()
    models = load_models(train_config, checkpoint_dir, cats)

    for cat_id, cat in enumerate(cats):
        print(cat)
        model = UNet()
        model.load_state_dict(models[cat]['model_state_dict'])

        for i in range(len(dataset)):
            data = dataset[i]
            image, _ = return_org_image_and_label_only(data, cat_id)
            output_segs = model(image.permute(0, 3, 1, 2).float()) # Handle through a function
            output_segs = output_segs.squeeze(0).permute(1, 2, 0)
            output_segs_numpy = output_segs.cpu().detach().numpy()

            # confusion: lot's of unique values then why the black and white only?
            output_segs_numpy_normalized = (output_segs_numpy - np.min(output_segs_numpy)) / (np.max(output_segs_numpy) - np.min(output_segs_numpy))
            image_numpy = image.squeeze(0).cpu().detach().numpy()
            visual_segments(type_='overlay', segments=output_segs_numpy_normalized, image=image_numpy, window_name=f"CATEGORY = {cat}")

if __name__ == '__main__':
    main()