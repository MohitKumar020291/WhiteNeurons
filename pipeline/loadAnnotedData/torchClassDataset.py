import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Type, List, Union
from .helper import ( 
                read_coco_file,
                read_yaml_file
                )
from utils import image2array
import os
from PIL import Image
from pipeline.loadAnnotedData.helper import show_image


class SingleSegmentedImage:
    def __init__(self, image_json_file: Dict, images_json_file: Dict, train_dest: str = None, cats: list = None, type_="train"):
        """
        Initializes the dataset with data and corresponding labels.
        Args:
            json_file (Dict): coco segmentation file
            annotations: annotations for an image
            labels (list or torch.Tensor): The corresponding labels.
        """
        self.image_json_file = image_json_file
        self.file_name = None
        self.image_id = image_json_file["id"]
        self.images_json_file = images_json_file
        self.cats = cats or [cat['name'] for cat in images_json_file['categories']]
        self.train_dest = train_dest
        self.data = list()
        self.org_image = None
        self.type_ = type_
        assert self.cal_data()
    
    def cal_data(self):
        image_id = self.image_json_file["id"]
        for category_id, _ in enumerate(self.cats):
            metadata_image = read_coco_file(image_id, category_id, self.images_json_file, self.train_dest, show=False)
            self.data.append(metadata_image)
            self.file_name = self.file_name or self.data[-1][0]
            if self.org_image is None: assert(self.retrive_org_image()), "Failed to retrieve the org_image"
            # self.org_image = metadata_image[1]
        return True
    
    def retrive_org_image(self):
        annorted_data_folder_path = read_yaml_file("config/pipeline_config.yaml")['annoted_data_folder_path']

        resolved_path = os.path.expanduser(annorted_data_folder_path)
        directories = os.listdir(resolved_path)
        if self.type_ not in directories:
            raise ValueError(f"cannot retreive orginal image from {self.type_} folder as it do not exists")
        
        type__path = resolved_path + f"/{self.type_}"
        image_files = os.listdir(type__path)

        for image_file in image_files:
            image_file_split = image_file.split(".")
            if image_file_split[0] == self.file_name:
                self.org_image_dest = type__path + f"/{image_file}"
                self.org_image = Image.open(self.org_image_dest)
                self.org_image = image2array(self.org_image)
                self.org_image = np.transpose(self.org_image, axes=(2, 0, 1))
        return True

    # def __len__(self):
    #     return len(self.data)
    
    def __getitem__(self, cat_id):
        """
        Retrieves a single data sample and its label at the given index.
        Args:
            image_cat_id: [image_id: int, category_id: int]
        """
        image_file_name, org_image, masked_image_tensor, accumulated_mask, image_json_file, train_dest, annotations, image_dest = \
            self.data[cat_id]

        return image_file_name, org_image, masked_image_tensor, accumulated_mask, \
            image_json_file, train_dest, annotations, image_dest
 

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
    
    def __len__(self):
        return len(self.data)

    def add_segmented_image(self, segmented_image: "SingleSegmentedImage"):
        # Doubt: Should we really allow the addition of data?
        # If yes, find a modern way to update self.images_json_file
        raise NotImplemented("The function has not been implemented")

    def __getitem__(self, ids: Union[list, int, tuple]):
        from collections.abc import Iterable
        # Handle (image_id, category_id) or [image_id, category_id]
        if isinstance(ids, (list, tuple)) and len(ids) == 2:
            # This should also only pass the seg images
            image_id, cat_id = ids
            seg_image = self.data[image_id]
            data_tuple = seg_image[cat_id]
            return (*data_tuple, seg_image)
        
        # Handle single image_id
        elif isinstance(ids, int):
            return self.data[ids]

        # Handle list of image_ids
        elif isinstance(ids, Iterable):
            outputs = [] #(acc_mask, seg_image)
            cat_id = ids[-1]
            for image_id in ids[:-1]:
                # return [self.data[i] for i in ids]
                idxx = [image_id, cat_id]
                outputs.append(self[idxx])
            return outputs
        else:
            raise TypeError("Invalid index type")