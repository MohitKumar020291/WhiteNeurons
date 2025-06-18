import cv2
import os
import numpy as np


def SaveScreenShot(output_folder: str, mask: np.array, ss_taken: int) -> None:
    """Saves the frame into output folder"""
    os.makedirs(output_folder, exist_ok=True)
    ss_path = os.path.join(output_folder, f"mask_ss_{ss_taken}.png")
    cv2.imwrite(ss_path, mask)