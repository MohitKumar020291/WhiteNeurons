import torch
from ..helper import read_coco_file, show_image
from ..torchClassDataset import CustomSegmentationDataset



def test_custom_datset():
    image_id = 0
    category_id = 2
    image_tensor1, accumulated_mask1, json_file, train_dest = read_coco_file(image_id, category_id, show=False, in_test=True)

    dataset = CustomSegmentationDataset(json_file, category_id, train_dest=train_dest)
    image_tensor2, accumulated_mask2 = dataset[image_id]


    assert torch.allclose(image_tensor1, image_tensor2) == True
    assert torch.allclose(accumulated_mask1, accumulated_mask2) == True
    print(image_tensor1.shape, image_tensor2.shape)
    show_image(image=image_tensor2, fmt='CHW')

if __name__ == "__main__":
    test_custom_datset()