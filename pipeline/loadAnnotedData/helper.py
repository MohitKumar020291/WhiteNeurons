import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from utils import read_yaml_file

def show_image(image, fmt='HWC'):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        image = (image * 255).astype('uint8')

    if fmt=='CHW':
        image = np.transpose(image, (1, 2, 0))

    window_name = 'Segmentation Result (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1) & 0xFF  # Capture key press

        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

def getMetaDataAboutCocoFolder(config_file: str = 'config/pipeline_config.yaml'):
    """
        config: The folder containing the coco segmentation images
    """
    # read config_file
    annorted_data_folder_path = read_yaml_file(config_file)['annoted_data_folder_path']
    resolved_path = os.path.expanduser(annorted_data_folder_path)

    files = os.listdir(resolved_path)

    if 'train' not in files:
        raise Exception("train folder is not present")

    train_folder_files = os.listdir(resolved_path + "/train")

    coco_filename = None
    for file in train_folder_files:
        if '.coco.json' in file:
            coco_filename = file

    if coco_filename is None:
        raise Exception("No coco file")

    # Read json file using
    train_dest = resolved_path + '/train'
    coco_destiny = train_dest + f"/{coco_filename}"

    return train_dest, coco_destiny

def getSegmentsForAnImage(json_file, image_id, category_id) -> list[list[tuple[float]]]:
    """
    Args
        json_file: metadata about image
        image_id: just for sanity check
    return
        accumulated mask for the image for a category - 1 and 0 on the x, y
    """
    annotations = json_file['annotations']
    segments = list()
    for annotation in annotations:
        if annotation['image_id'] == image_id and annotation['category_id'] == category_id:
            segments.append(annotation['segmentation'][0])

    #x, y
    segments_xy = list()
    for segment_ in segments:
        segments_xy.append([])   
        for id, x in enumerate(segment_):
            if id % 2 == 0:
                segments_xy[-1].append((x, segment_[id+1]))
    
    return segments_xy, segments

def maskImageFromSegment(segments, image, fill=True):
    # mask for opencv
    # [(x1, y1), (x2, y2)...]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for seg in segments:
        pts = np.array(seg, np.int32).reshape((-1, 1, 2))
        if fill:
            cv2.fillPoly(image, [pts], color=(0, 255, 0))
            cv2.fillPoly(mask, [pts], color=1)
        else:
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return mask, image

def getImagesFromJson(json_file, image_id: int) -> str:
    images = json_file['images']
    for image in images:
        if image['id'] == image_id:
            return image

def read_json_file(coco_destiny):
    with open(coco_destiny, 'r') as file:
        json_file = json.load(file)
    return json_file

def read_coco_file(image_id, category_id, images_json_file, train_dest, show: bool = True):
    """
    returns
        image: a cv2.imread object
        image_tensor: a tensor version of this image which shape CHW
        mask_tensor: a tensor which represents the mask of the categor 
        (0 or 1 for each pixel), shape = 1HW
    """
    image_json_file = getImagesFromJson(images_json_file, image_id)

    # Loading categories
    cats = []
    for cat in images_json_file['categories']:
        cats.append(cat['name'])

    # image_name = getImagesFromJson(json_file, image_id)
    image_name = image_json_file["file_name"]
    image_dest = train_dest + f'/{image_name}'
    image = cv2.imread(image_dest)

    # Only pass the json_file for an image
    segments_xy, annotations = getSegmentsForAnImage(images_json_file, image_id, category_id)
    accumulated_mask, image = maskImageFromSegment(segments=segments_xy, image=image)
    if show:
        show_image(image)
    
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(accumulated_mask).unsqueeze(0).float()

    return image, image_tensor, mask_tensor, image_json_file, annotations, image_dest

def CreateCollectionOfSegmentatedImages():
    """
    The function provides the CollectionOfSegmentatedImages
    by default the data is taken from 'config/pipeline_config.yaml' or 
    """
    from .torchClassDataset import CollectionOfSegmentatedImages

    train_dest, coco_destiny = getMetaDataAboutCocoFolder()
    images_json_file = read_json_file(coco_destiny)
    cats = [cat['name'] for cat in images_json_file['categories']]
    cosi = CollectionOfSegmentatedImages(
                    images_json_file, 
                    train_dest,
                    cats=cats
                    )
    return cosi



if __name__ == "__main__":
    read_coco_file()
