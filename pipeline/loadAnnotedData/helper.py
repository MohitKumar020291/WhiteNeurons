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

    if image is not None:
        while True:
            cv2.imshow('image', image)

            key = cv2.waitKey(1) & 0xFF  # Capture key press

            if key == ord('q'):
                break

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
    annotations = json_file['annotations']
    segments = []
    for annotation in annotations:
        if annotation['image_id'] == image_id and annotation['category_id'] == category_id:
            segments.append(annotation['segmentation'][0])
    
    #x, y
    segments_xy = []
    for segment_ in segments:
        segments_xy.append([])   
        for id, x in enumerate(segment_):
            if id % 2 == 0:
                segments_xy[-1].append((x, segment_[id+1]))
    
    return segments_xy

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
            image_name = image['file_name']
            return image_name

def read_json_file(coco_destiny):
    with open(coco_destiny, 'r') as file:
        json_file = json.load(file)
    return json_file

def read_coco_file(image_id, category_id, show=True, in_test=False):
    train_dest, coco_destiny = getMetaDataAboutCocoFolder()

    json_file = read_json_file(coco_destiny)

    # Loading categories
    cats = []
    for cat in json_file['categories']:
        cats.append(cat['name'])

    image_name = getImagesFromJson(json_file, image_id)
    image_dest = train_dest + f'/{image_name}'
    image = cv2.imread(image_dest)

    segments_xy = getSegmentsForAnImage(json_file, image_id, category_id)
    # Just taking the first mask, how to combine multiple mask: there could be
    # May be this is the way, but the boundaries will be continuos in this
    # combined_segments_xy = []
    # for segment_xy_ in segments_xy:
    #     combined_segments_xy.extend(segment_xy_)
    # accumulated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # for mask in segments_xy:
    #     pts, image = maskImageFromSegment(mask, image)
    #     cv2.fillPoly(accumulated_mask, [pts], color=1)
    # show_image(image)
    accumulated_mask, image = maskImageFromSegment(segments=segments_xy, image=image)
    if show:
        show_image(image)
    
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(accumulated_mask).unsqueeze(0).float()

    if in_test == True:
        return image_tensor, mask_tensor, json_file, train_dest
    return image_tensor, mask_tensor

if __name__ == "__main__":
    read_coco_file(0, 2)
