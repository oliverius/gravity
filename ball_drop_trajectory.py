import cv2
from manual_frame_selection import ManualFrameSelection

def get_trajectory_in_pixels(
    video_path: str,
    rotate_video: bool,
    start_frame:int,
    end_frame:int, fps:int) -> tuple[list, list]:
    # Important that we keep the scale 100% or we will get less points
    # This is due to us defining the size of the ball > 50px but if scaled down the ball is also smaller
    player = ManualFrameSelection(video_path, rotate_video, 1)

    frame_idx     = 0
    relative_idx  = 0  # Used for timing

    min_contour_area = 50 # px

    positions_px  = []
    times         = []

    cap = player.open_file()
    player.output_video_information(cap)

    while cap.isOpened():
        frame = player.get_frame(cap)
        if frame is None:
            break
        
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if frame_idx > end_frame:
            break

        mask = player.create_red_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > min_contour_area:
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cy = int(M['m01'] / M['m00'])  # Vertical center
                    positions_px.append(cy)
                    times.append(relative_idx / fps)

        frame_idx += 1
        relative_idx += 1

    cap.release()
    return positions_px, times