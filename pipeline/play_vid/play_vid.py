import cv2
from utils import draw_bounding_box

def play_vid(frame, mask, show_by):
    if show_by == 'frame':
        frame_with_box = draw_bounding_box(frame, mask)
        cv2.imshow('frame', frame_with_box)
    elif show_by == 'mask':
        cv2.imshow('frame', mask)