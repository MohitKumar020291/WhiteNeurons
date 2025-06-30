import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Type, List, Union
from .helper import ( 
                read_coco_file,
                )


class SingleSegmentedImage:
    def __init__(self, image_json_file: Dict, images_json_file: Dict, train_dest: str = None, cats: list = None):
        """
        Initializes the dataset with data and corresponding labels.
        Args:
            json_file (Dict): coco segmentation file
            annotations: annotations for an image
            labels (list or torch.Tensor): The corresponding labels.
        """
        self.image_json_file = image_json_file
        self.image_id = image_json_file["id"]
        self.images_json_file = images_json_file
        self.cats = cats or [cat['name'] for cat in images_json_file['categories']]
        self.train_dest = train_dest
        self.data = list()
        assert self.cal_data()
    
    def cal_data(self):
        image_id = self.image_json_file["id"]
        for category_id, _ in enumerate(self.cats):
            self.data.append(read_coco_file(image_id, category_id, self.images_json_file, self.train_dest, show=False))
        return True

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, cat_id):
        """
        Retrieves a single data sample and its label at the given index.
        Args:
            image_cat_id: [image_id: int, category_id: int]
        Returns:
            tuple: A tuple containing the data sample and its label.
        """
        image, image_tensor, accumulated_mask, image_json_file, train_dest, annotations = \
            self.data[cat_id]

        return image, image_tensor, accumulated_mask, \
            image_json_file, train_dest, annotations
 

class CollectionOfSegmentatedImages(Dataset):
    segmented_images: List[Type["SingleSegmentedImage"]]
    def __init__(self, images_json_file: Dict, train_dest: str, cats: list = None):
        """
        Args
            json_files: list og json_file from 
        """
        self.images_json_file = images_json_file
        self.train_dest = train_dest
        self.data = dict()
        self.cats = cats or [cat['name'] for cat in self.images_json_files[0]['categories']]
        assert self.use_json_files(), "Error in use_json_file"
    
    def use_json_files(self):
        # Sanity checks -- Implement
        # check if the format is coco file - could be extended to other files also

        for image_json_file in self.images_json_file["images"]:
            self.data[image_json_file["id"]] = \
            SingleSegmentedImage(
                    image_json_file, 
                    self.images_json_file,
                    train_dest=self.train_dest,
                    cats=self.cats
                    )
        return True

    def add_segmented_image(self, segmented_image: "SingleSegmentedImage"):
        raise NotImplemented("The function has not been implemented")
        # Doubt: Should we really allow the addition of data?
        # If yes, find a modern way to update self.images_json_file
        if not isinstance(segmented_image, "SegmentedImage"):
            raise ValueError(f"segmented image should be of class SegmentedImage, provided {type(segmented_image)}")
        
        # A warning if the image is already present
        self.data[segmented_image.image_id] = segmented_image
        self.images_json_file.append(segmented_image.image_json_file)
        return
    
    def __getitem__(self, ids: Union[list, int]):
        """
        ids:
            if int - then considered as image id
            if list -  then considered as image id, category id 
        returns:
            dict[image_id] = [accumulated masks for each category_i]
        """
        cat_id = None
        if isinstance(ids, list):
            image_id = ids[0]
            cat_id = ids[1]
        else:
            image_id = ids

        segmented_image = self.data[image_id]
        acc_masks = dict()
        for cat_id_, _ in enumerate(self.cats):
            acc_masks[cat_id_] = segmented_image[cat_id_]
        
        if cat_id:
            return acc_masks[cat_id]
        return acc_masks
