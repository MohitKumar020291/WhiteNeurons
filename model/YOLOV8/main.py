import os
import yaml
from typing import Tuple
from ultralytics import YOLO



def read_yaml_file(filepath):
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError as e:
        raise
    except yaml.YAMLError as e:
        raise
    except Exception as e:
        raise


def init_train() -> Tuple[str, str]:
    # Getting current path
    current_dir = os.path.dirname(__file__)
    config_file_name = "config.yaml"
    config_file_path = os.path.join(current_dir, config_file_name)
    assert os.path.exists(config_file_path), f"config_yaml could not be located at {config_file_path}"

    return config_file_path, current_dir


def main() -> None:
    config_file_path, current_dir = init_train()
    assert os.path.exists(config_file_path), f"Config file not found at {config_file_path}"

    # model = YOLO("yolo11s-seg.pt")
    # model = YOLO("yolo11n-seg.pt") # error
    model = YOLO("yolo11m-seg.pt") 
    print("model loaded") 
    results = model.train(
        data=config_file_path, 
        epochs=32, 
        imgsz=640,
        project=current_dir,
        name="train"
        )



if __name__ == "__main__":
    main()
