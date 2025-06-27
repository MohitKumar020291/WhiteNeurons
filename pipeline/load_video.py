import cv2
from utils import metaDataVideo


def load_video(
        input_file,
        step
):
    """Captures device and returns metaData about video"""
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise Exception(f"Could not open the video file: {input_file}")
    step_frames, max_ss_count = metaDataVideo(input_file, step=step)
    return cap, step_frames, max_ss_count