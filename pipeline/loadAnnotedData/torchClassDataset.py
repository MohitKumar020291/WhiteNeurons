import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict
from .helper import (getImagesFromJson, 
                read_coco_file,
                getSegmentsForAnImage,
                maskImageFromSegment,
                show_image)

class CustomSegmentationDataset(Dataset):
    def __init__(self, json_file: Dict, category_id: str, train_dest: str = None):
        """
        Initializes the dataset with data and corresponding labels.
        Args:
            json_file (Dict or str): coco segmentation file or file path
            labels (list or torch.Tensor): The corresponding labels.
        """
        if isinstance(json_file, str):
            self.json_file = read_coco_file(json_file)
        else:
            self.json_file = json_file
        self.category_id = category_id
        self.train_dest = train_dest
        self.data = self.json_file["images"]

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self: "CustomSegmentationDataset", image_id: int):
        """
        Retrieves a single data sample and its label at the given index.
        Args:
            idx (int): The index of the image to retrieve.
        Returns:
            tuple: A tuple containing the data sample and its label.
        """
        image_info = self.data[image_id]
        image_id = image_info["id"]
        image_name = image_info["file_name"]
        image_path = f"{self.train_dest}/{image_name}"

        image = cv2.imread(image_path)

        segments_xy = getSegmentsForAnImage(self.json_file, image_id, self.category_id)

        # accumulated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # for seg in segments_xy:
        #     pts, image = maskImageFromSegment(seg, image)
        #     # These points get filled with color at pts
        #     cv2.fillPoly(accumulated_mask, [pts], color=1)
        accumulated_mask, image = maskImageFromSegment(segments=segments_xy, image=image)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(accumulated_mask).unsqueeze(0).float()

        return image_tensor, mask_tensor
    