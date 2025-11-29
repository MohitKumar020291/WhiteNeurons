import os
from typing import Tuple
from ultralytics import YOLO

from utils import read_yaml_file


def init_train() -> Tuple[str, str]:
    # Getting current path
    current_dir = os.path.dirname(__file__)
    config_file_name = "config.yaml"
    config_file_path = os.path.join(current_dir, config_file_name)

    return config_file_path, current_dir

def main() -> None:
    config_file_path, current_dir = init_train()
    assert os.path.exists(config_file_path), f"Config file not found at {config_file_path}"

    model = YOLO("yolo11n-seg.pt")
    results = model.train(
        data=config_file_path, 
        epochs=16, 
        imgsz=640,
        project=current_dir,
        name="train"
        )



if __name__ == "__main__":
    main()