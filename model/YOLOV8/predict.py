import os
import cv2

from ultralytics import YOLO
from model.YOLOV8.main import read_yaml_file



current_dir = os.path.dirname(__file__) # trained model is there
model_rel_path = 'train/weights/last.pt'
print(os.path.join(current_dir, model_rel_path))
print(os.path.exists(os.path.join(current_dir, model_rel_path)))
model_path = os.path.join(current_dir, model_rel_path)
assert os.path.exists(model_path)

config = read_yaml_file('config.yaml')

yolov8_path = config['path'][3:]
yolov8_path_abs = os.path.join(current_dir, yolov8_path)
assert os.path.exists(yolov8_path_abs)

test_path = config['test']
test_path_abs = os.path.join(yolov8_path_abs, test_path)
assert os.path.exists(test_path_abs)

test_image_name = 'mask_ss_102_png.rf.fc5be69df826fd0e1dbb69bf615a3036.jpg'
test_image_path_abs = os.path.join(test_path_abs, test_image_name)
assert os.path.exists(test_image_path_abs)


img = cv2.imread(test_image_path_abs)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.numpy() * 255
        mask = cv2.resize(mask, (W, H))
        cv2.imwrite('./output.png', mask)
