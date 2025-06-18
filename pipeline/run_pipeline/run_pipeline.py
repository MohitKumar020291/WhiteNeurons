import cv2
import os

from ..load_video import load_video
from ..apply_mask import apply_mask
from ..save_screenshot import SaveScreenShot
from ..refine_mask import refine_mask
from ..play_vid import play_vid

def run(
        input_file,
        step=0.2,
        output_folder='./ss',
        color=[0, 0, 0],
        show_by='frame',
        use_binary_mask=False,
        take_ss=False,
        ):
    
    # Possible but the frames are shifting one-one so the save might work one at a time
    # if parallel:
    #     Parallel(n_jobs=4)(
    #         delayed(SaveScreenShot)(output_folder, mask, idx) for idx, mask in enumerate(masks)
    #     )

    cap, step_frames, max_ss_count = load_video(input_file, step)
    ss_taken = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, hsvImage,  = apply_mask(frame, use_binary_mask, show_by, color)

        # Apply V-channel refinement for HSV-based mask only
        # if show_by == 'mask' and use_binary_mask and hsvImage is not None:
        if show_by == 'mask' and (
            (use_binary_mask and hsvImage is not None) or (not use_binary_mask and hsvImage is not None)
        ):
            mask = refine_mask(mask, hsvImage, v_threshold=100)

        if take_ss and show_by == 'mask' and frame_number % step_frames == 0 and ss_taken < max_ss_count:
            SaveScreenShot(output_folder, mask, ss_taken)
            ss_taken += 1
            
        play_vid(frame, mask, show_by)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()