import os
from typing import Tuple
from ultralytics import YOLO

from utils import read_yaml_file

class YOLOTrain:
    def __init__(self):
        self.current_dir = os.path.dirname(__file__)
        self.config_file_name = "config.yaml"
        self.config_file_path = os.path.join(self.current_dir, self.config_file_name)
        assert os.path.exists(self.config_file_path), f"Config file not found at {self.config_file_path}"

    def train_model(self):
        assert os.path.exists(self.config_file_path), f"Config file not found at {self.config_file_path}"

        model = YOLO("yolo11n-seg.pt")
        results = model.train(
            data=self.config_file_path, 
            epochs=1, 
            imgsz=640,
            project=self.current_dir,
            name="train"
        )


if __name__ == "__main__":
    # main()
    yolov8 = YOLOTrain()
    yolov8.train_model()