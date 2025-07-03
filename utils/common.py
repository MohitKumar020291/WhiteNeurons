import numpy as np
import cv2
from PIL import Image
import yaml
from typing import Callable


def get_binary_mask(frame, threshold=180):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def get_hsv_mask(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = get_limits(color=color)
    mask = cv2.inRange(hsv, lower, upper)
    return mask, hsv


def get_hsv_from_masked_region(frame, mask):
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    hsv_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    return hsv_masked


def refine_mask_with_value(mask, hsv, v_threshold=60):
    value = hsv[:, :, 2]
    mask[value > v_threshold] = 0
    return mask


def draw_bounding_box(frame, mask):
    pil_mask = Image.fromarray(mask)
    bbox = pil_mask.getbbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame


def get_limits(color, tol_h=15, tol_s=80, tol_v=80):
    color = np.uint8([[color]])  # Convert BGR to a 1x1 image
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]

    lower_limit = np.array([
        max(hsv_color[0] - tol_h, 0),
        max(hsv_color[1] - tol_s, 0),
        max(hsv_color[2] - tol_v, 0)
    ])
    upper_limit = np.array([
        min(hsv_color[0] + tol_h, 200),
        min(hsv_color[1] + tol_s, 255),
        min(hsv_color[2] + tol_v, 255)
    ])
    return lower_limit, upper_limit


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
    

def metaDataVideo(input_file: str, step: 0.2):
    temp_video = cv2.VideoCapture(input_file)
    if not temp_video.isOpened():
        raise Exception("Could not open the video file.")

    frame_rate = temp_video.get(cv2.CAP_PROP_FPS)
    frame_count = temp_video.get(cv2.CAP_PROP_FRAME_COUNT)
    video_length_in_secs = frame_count / frame_rate
    step_frames = int(step * frame_rate)
    max_ss_count = int(1 / step * video_length_in_secs)  # total screenshots to take
    temp_video.release()

    return step_frames, max_ss_count

def image2array(image) -> np.ndarray:
  return np.array(image)

def bench(fn: Callable, args: dict) -> None:
    import time
    start_time = time.time()
    fn(**args)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"FUNCTION: {fn} \n ARGS: {args} \n TIME_TAKE: {time_taken}")
    return time_taken