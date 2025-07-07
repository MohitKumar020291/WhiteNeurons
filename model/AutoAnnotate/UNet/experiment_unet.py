import os

from model.AutoAnnotate.helper import load_models, test_init, post_process_infer, read_yaml_file, show_image
from model.AutoAnnotate.UNet.function_blocks import UNet
from model.AutoAnnotate.AutoAnnotate.seg_unet import loading_images, get_cats



def get_model(cat: str, cats: list[str]):
    assert cat in cats, f"cat should be one of {cats}"

    model = UNet()
    # read model from checkpoints
    unet_directory = "/home/mohitb1i/pytorch_env/WhiteNeurons/model/AutoAnnotate/UNet" # hardcoded
    train_config, checkpoint_dir = test_init(unet_directory)
    models = load_models(train_config, checkpoint_dir, cats)
    # print(models['hand']['model_state_dict'].keys())

    model.load_state_dict(models[cat]['model_state_dict'])
    return model


def main():
    current_directory = os.path.dirname(__file__)
    images = loading_images(current_directory)

    cat = 'noise'
    cats = get_cats()
    model = get_model(cat, cats)

    crt_cls_imgs_idx_yaml_path = os.path.join(current_directory, 'crt_cls_imgs_idx.yaml')
    crt_cls_imgs_idx_yaml = read_yaml_file(crt_cls_imgs_idx_yaml_path)
    crt_cls_imgs_c_idx = crt_cls_imgs_idx_yaml['c_idx']
    crt_cls_imgs_ic_idx = crt_cls_imgs_idx_yaml['ic_idx']
    crt_cls_imgs_mid_idx = crt_cls_imgs_idx_yaml['mid_idx']

    idxs = [crt_cls_imgs_c_idx, crt_cls_imgs_ic_idx, crt_cls_imgs_mid_idx]
    work_idx = 0
    print(f"MODEL: {cat}")
    for idxx in idxs[work_idx]:
        image = images[idxx].unsqueeze(0)
        output_segs = model(image.permute(0, 3, 1, 2).float()) # Handle through a function
        image_numpy, output_segs_numpy_normalized = post_process_infer(image, output_segs)
        show_image(image_numpy)
        show_image(output_segs_numpy_normalized)


            


if __name__ == "__main__":
    main()