import os
import torch.nn as nn
from collections import OrderedDict

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

    model.load_state_dict(models[cat]['model_state_dict'])
    return model, models


def get_model_layer(
                top_model,
                module: list[str], 
                prev_state_dict
                ):
    layer_model = top_model()
    layer_name_list = module.split('.')
    for layer in layer_name_list:
        if layer not in not_a_model:
            try:
                layer_int = int(layer)
                layer_model = layer_model[layer_int]
            except:
                layer_model = layer_model._modules[layer]
        else:
            break

    layer_model_state_dict = dict()
    prefix = module + '.'
    for state, value in prev_state_dict.items():
        if state.startswith(prefix):
            new_key = state[len(prefix):]
            layer_model_state_dict[new_key] = value
    
    # loading individual model with it's states
    print(module, layer_model_state_dict.keys())
    return
    layer_model.load_state_dict(layer_model_state_dict)
    return layer_model


def get_model_layers(top_model, model_state_dict, layer_name: str):   
    global not_a_model
    not_a_model = ['weight', 'bias', 'running_mean', 'running_var'] # Will update this list
    layer_model = UNet()
    prev_state_dict = dict()
    model_state_dict_keys = list(model_state_dict.keys())
    for idx, model_state_key in enumerate(model_state_dict_keys):
        if layer_name in model_state_dict_keys[idx] and idx < len(model_state_dict_keys) - 1 \
            and layer_name not in model_state_dict_keys[idx+1]:
            prev_state_dict[model_state_key] = model_state_dict[model_state_key]
            break
        else:
            prev_state_dict[model_state_key] = model_state_dict[model_state_key]

    # Squashing
    prev_modules = list()
    prev_model_state_keys = prev_state_dict.keys()
    for model_state_key in prev_model_state_keys:
        model_state_keys = model_state_key.split(".")
        for idx, key in enumerate(model_state_keys):
            if key in not_a_model:
                layer_model = ""
                for model_state_key in model_state_keys[:idx]:
                    layer_model += "."
                    layer_model += model_state_key
                if layer_model[1:] not in prev_modules:
                    prev_modules.append(layer_model[1:])

    ordered_list_modules = OrderedDict()
    # ordered_list_modules = []
    for module in prev_modules:
        # Module name cannot contain .
        ordered_list_modules["".join(module.split("."))] = get_model_layer(top_model, module, prev_state_dict)

    model = nn.Sequential(ordered_list_modules)

    return model
    


def get_imgs_idx(current_directory: str):
    crt_cls_imgs_idx_yaml_path = os.path.join(current_directory, 'crt_cls_imgs_idx.yaml')
    crt_cls_imgs_idx_yaml = read_yaml_file(crt_cls_imgs_idx_yaml_path)
    crt_cls_imgs_c_idx = crt_cls_imgs_idx_yaml['c_idx']
    crt_cls_imgs_ic_idx = crt_cls_imgs_idx_yaml['ic_idx']
    crt_cls_imgs_mid_idx = crt_cls_imgs_idx_yaml['mid_idx']

    idxs = [crt_cls_imgs_c_idx, crt_cls_imgs_ic_idx, crt_cls_imgs_mid_idx]

    return idxs


def transformed_image(layer_model, image):
    return layer_model(image)


def main():
    # current_directory = os.path.dirname(__file__)
    # images = loading_images(current_directory)

    cat = 'noise'
    cats = get_cats()
    model, models = get_model(cat, cats)

    model_state_dict = models[cat]['model_state_dict']
    layer_name = 'enc3.conv.double_conv.0'
    model = get_model_layers(UNet, model_state_dict, layer_name) # layer_model

    # idxs = get_imgs_idx(current_directory)

    # work_idx = 0
    # print(f"MODEL: {cat}")
    # for idxx in idxs[work_idx]:
    #     image = images[idxx].unsqueeze(0)
    #     output_segs = model(image.permute(0, 3, 1, 2).float()) # Handle through a function
    #     image_numpy, output_segs_numpy_normalized = post_process_infer(image, output_segs)
    #     show_image(image_numpy)
    #     show_image(output_segs_numpy_normalized)



# Extend this code to process the UNet Class such that I can infer from it


if __name__ == "__main__":
    main()