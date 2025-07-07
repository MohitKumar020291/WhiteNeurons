import os
import pickle
import torch
import cv2
from numpy import ndarray as npnda
import numpy as np
from typing import Union, Tuple, Dict, AnyStr
from skimage.color import label2rgb
from pipeline.loadAnnotedData.torchClassDataset import SingleSegmentedImage

# modules are running individually
# python3 -m model.AutoAnnotate.tests.segment_test
from pipeline.loadAnnotedData.helper import show_image
from utils import image2array, read_yaml_file




def visual_segments(type_: str, segments: Union[npnda, str], image, show=True, window_name=None):
    """
    Args:
        - type_
            - org: shows the real b/w image - normalized version
            - color: color codes the org image
            - overlay: overlays over the real image
        - image: either real image array or path to read image from
    """
    types = ['org', 'color', 'overlay']
    assert type_ in types, f"type_ should belong to {types}"
    if type_ == 'overlay' and image is None:
        raise Exception(f"The original image, image is required, provided {image}")
    
    from numpy import squeeze, uint8
    image = image2array(image)
    if image.dtype != uint8:
        image = (image * 255).clip(0, 255).astype(uint8)
    segments = squeeze(segments, axis=-1)
    if len(image.shape) == 2:  # grayscale to 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    segments_normalized = (segments * 255 / segments.max()).astype('uint8')
    # segments_colored = cv2.applyColorMap(segments_normalized, cv2.COLORMAP_JET)
    segments_colored = label2rgb(segments, image=image, bg_label=0) # maps each label (region id) to a different colors
    segments_overlayed = cv2.addWeighted(
        image, 0.7,
        cv2.applyColorMap(
            segments_normalized,
            cv2.COLORMAP_JET
        ), 0.3, 0
    )

    if show:
        if type_ == 'org':
            show_image(segments_normalized, window_name=window_name)
        elif type_ == 'color':
            show_image(segments_colored, window_name=window_name)
        else:
            show_image(segments_overlayed, window_name=window_name)
    else:
        images = [segments_normalized, segments_colored, segments_overlayed]
        return images[types.index(type_)]
    

def visual_image(image) -> None:
    image = image2array(image) if not isinstance(image, npnda) else image
    show_image(image)
    return None


def return_org_image_and_label_only(datas, cat_id=None):
    import torch
    import numpy as np

    # This will loop over the categories
    if isinstance(datas, SingleSegmentedImage):
        if cat_id is None:
            raise ValueError("The cat_id could not be None in case of a single instance of SingleSegmentedImage")        
        data = datas[cat_id]
        return return_org_image_and_label_only([data])

    images = []
    labels = []
    for data in datas:
        # This loop have do not contain the segmented_images, read CollectionOfSegmentatedImages
        images.append(data[1])  # org image: [H, W, C] or [C, H, W]
        labels.append(data[3])  # accumulated mask: [H, W, 1]

    images = torch.tensor(np.stack(images))  # [N, H, W, C]
    labels = torch.tensor(np.squeeze(np.stack(labels), axis=-1))  # [N, H, W]

    assert (images.shape[:-1] == labels.shape)

    # For Conv2D
    return images, labels.unsqueeze(1).float()


def test_init(unet_directory) -> Tuple[Dict, AnyStr]:
    # current_directory = os.path.dirname(__file__)
    train_config_path = os.path.join(unet_directory, 'unet_config.yaml')
    train_config = read_yaml_file(train_config_path)

    checkpoint_dir = train_config.get("checkpoint_dir")
    if checkpoint_dir is None:
        print(f"provide the checkpoints dir, where models are stored!, provided = {checkpoint_dir}")
    checkpoint_dir = os.path.join(unet_directory, checkpoint_dir)
    
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



def cache_if_not_exists(path, compute_fn, update=False):
    if not os.path.exists(path) or update:
        print("Cache not found or update requested. Recomputing...")
        result = compute_fn()
        with open(path, 'wb') as f:
            pickle.dump(result, f)
    else:
        print("Loading from cache:", path)
        with open(path, 'rb') as f:
            result = pickle.load(f)
    return result



def post_process_infer(image, output_segs):
    output_segs = output_segs.squeeze(0).permute(1, 2, 0)
    output_segs_numpy = output_segs.cpu().detach().numpy()

    # confusion: lot's of unique values then why the black and white only?
    output_segs_numpy_normalized = (output_segs_numpy - np.min(output_segs_numpy)) / (np.max(output_segs_numpy) - np.min(output_segs_numpy))
    image_numpy = image.squeeze(0).cpu().detach().numpy()

    return image_numpy, output_segs_numpy_normalized