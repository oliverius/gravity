import cv2
import numpy as np

class ManualFrameSelection:
  def __init__(self, video_path: str, rotate_video: bool, scale: int):
      self.video_path = video_path
      self.rotate_video = rotate_video
      self.scale = scale
  
  def get_frame(self, cap:cv2.VideoCapture):
      ret, frame = cap.read()

      if not ret:
          print("End of video.")
          return None

      if self.rotate_video:
          frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
      
      if self.scale != 1:
          frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)

      return frame
  
  def open_file(self) -> cv2.VideoCapture | None:
      cap = cv2.VideoCapture(self.video_path)
      if cap.isOpened():
          return cap
      else:
          print("Could not open video file. Check the path or format.")
          return None

  def output_frame_selection(self, start_frame: int, end_frame: int) -> None:
      print(f"\nSelected frames → Start: {start_frame}, End: {end_frame}")

  def output_video_information(self, cap: cv2.VideoCapture) -> None:
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      fps = cap.get(cv2.CAP_PROP_FPS)
      print(f"Total frames: {total_frames}, FPS: {fps}")

  def get_red_colour_mask_values(self) -> tuple:
      # Red color mask (tuned for your ball). Two ranges because of how HSV colour space works
      lower_red1 = np.array([0, 120, 120])
      upper_red1 = np.array([10, 255, 255])
      lower_red2 = np.array([170, 120, 120])
      upper_red2 = np.array([180, 255, 255])
      return lower_red1, upper_red1, lower_red2, upper_red2
  
  def create_red_mask(self, frame):
      lower_red1, upper_red1, lower_red2, upper_red2 = self.get_red_colour_mask_values()
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
      mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
      mask = cv2.bitwise_or(mask1, mask2)
      return mask

  def overlay_text(self, blended, frame_idx: int, start_frame: int, end_frame: int) -> None:
      text_scale = 0.4
      cv2.putText(blended, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 1)

      if start_frame is not None:
          cv2.putText(blended, f"Start: {start_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 1)
      
      if end_frame is not None:
          cv2.putText(blended, f"End: {end_frame}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 1)

      cv2.imshow('Frame + Transparent Mask (s=start, e=end, q=quit)', blended)

  
  def play_frame_and_transparent_mask(self) -> None:
    cap = self.open_file()
    if cap is None:
        return
    self.output_video_information(cap)

    frame_idx = 0
    start_frame = None
    end_frame = None
    
    while cap.isOpened():
      frame = self.get_frame(cap)
      if frame is None:
        break
      
      mask = self.create_red_mask(frame)

      # === Make mask red and blend with frame
      mask_colored = cv2.merge([mask * 0, mask * 0, mask])  # blue=0, green=0, red=mask
      blended = cv2.addWeighted(frame, 1.0, mask_colored.astype(np.uint8), 0.4, 0)

      self.overlay_text(blended, frame_idx, start_frame, end_frame)
  
      key = cv2.waitKey(0) & 0xFF
      if key == ord('q'):
        break
      elif key == ord('s'):
        start_frame = frame_idx
        print(f"Start frame set to {start_frame}")
      elif key == ord('e'):
        end_frame = frame_idx
        print(f"End frame set to {end_frame}")

      frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

video = ManualFrameSelection("videos/ball drop.mp4", True, 0.5)
video.play_frame_and_transparent_mask()