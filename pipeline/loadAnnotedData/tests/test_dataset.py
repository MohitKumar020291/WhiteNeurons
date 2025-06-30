import torch
from ..helper import (read_coco_file, 
                      show_image, 
                      getMetaDataAboutCocoFolder, 
                      read_json_file,
                      CreateCollectionOfSegmentatedImages)
from ..torchClassDataset import SingleSegmentedImage


def test_segmented_image():
    """
    This is a test for SegmentedImage
    """
    image_id = 0
    category_id = 3
    train_dest, coco_destiny = getMetaDataAboutCocoFolder()
    images_json_file = read_json_file(coco_destiny)

    _, image_tensor1, accumulated_mask1, image_json_file, _, image_dest = \
        read_coco_file(image_id, category_id, images_json_file, train_dest, show=False)

    seg_image = SingleSegmentedImage(
                    image_json_file, images_json_file,
                    train_dest=train_dest
                    )

    # pulling out the image, accumulated mask for that category
    _, image_tensor2, accumulated_mask2, _, _, _ = seg_image[3]

    assert torch.allclose(image_tensor1, image_tensor2) == True
    assert torch.allclose(accumulated_mask1, accumulated_mask2) == True
    show_image(image=image_tensor2, fmt='CHW')
    

def test_custom_datset():
    """
    This is a visual test
    """
    image_id = 0
    category_id = 3

    train_dest, coco_destiny = getMetaDataAboutCocoFolder()
    images_json_file = read_json_file(coco_destiny)
    _, image_tensor1, accumulated_mask1, _, _, image_dest = \
        read_coco_file(image_id, category_id, images_json_file, train_dest=train_dest, show=False)

    # dataset = CustomSegmentationDataset(json_file, category_id, train_dest=train_dest)
    dataset = CreateCollectionOfSegmentatedImages()

    _, image_tensor2, accumulated_mask2, _, _, image_dest = \
            dataset[[image_id, category_id]]

    assert torch.allclose(image_tensor1, image_tensor2) == True
    assert torch.allclose(accumulated_mask1, accumulated_mask2) == True
    print("Test successful")
    show_image(image=image_tensor2, fmt='CHW')



if __name__ == "__main__":
    # test_segmented_image()
    test_custom_datset()