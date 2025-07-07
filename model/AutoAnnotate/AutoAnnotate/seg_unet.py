import torch
import numpy as np
import sys, os

from pipeline.load_ss import load_ss
from model.AutoAnnotate.UNet.dataset_loading import load_train_data
from model.AutoAnnotate.helper import cache_if_not_exists
from model.AutoAnnotate.helper import test_init, load_models, post_process_infer
from model.AutoAnnotate.UNet.function_blocks import UNet
from model.AutoAnnotate.helper import visual_segments, show_image




# Use less for now
def get_cats():
    val_dataset = load_train_data(type_="valid")
    seg_image = val_dataset[0]
    return seg_image.cats


def loading_images(current_directory):
    update_cache = False
    for arg in sys.argv[1:]:
        if arg.startswith("update_cache="):
            val = arg.split("=")[1].lower()
            if val in ["true", "1", "yes"]:
                update_cache = True

    print("Update Cache:", update_cache)

    torch.set_default_device("cuda")

    cache_path = os.path.join(current_directory, "image_cache.pkl")

    images = cache_if_not_exists(
        path=cache_path,
        compute_fn=lambda: torch.tensor(np.stack(load_ss(False))),
        update=update_cache
    )

    return images


def main():
    current_directory = os.path.dirname(__file__)
    images = loading_images(current_directory)

    cats = get_cats()
    unet_directory = "/home/mohitb1i/pytorch_env/WhiteNeurons/model/AutoAnnotate/UNet" # hardcoded
    train_config, checkpoint_dir = test_init(unet_directory)

    models = load_models(train_config, checkpoint_dir, cats)
    for cat in cats[2:]:
        print("CATEGORY:", cat)
        model = UNet()
        model.load_state_dict(models[cat]['model_state_dict'])
        for image in images:
            image = image.unsqueeze(0)
            output_segs = model(image.permute(0, 3, 1, 2).float()) # Handle through a function
            image_numpy, output_segs_numpy_normalized = post_process_infer(image, output_segs)
            show_image(image_numpy)
            show_image(output_segs_numpy_normalized)
            visual_segments(type_='overlay', segments=output_segs_numpy_normalized, image=image_numpy, window_name=f"CATEGORY = {cat}")
        


if __name__ == "__main__":
    main()