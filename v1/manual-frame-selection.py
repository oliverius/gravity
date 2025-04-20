import cv2
import numpy as np

video_path = 'videos/ball drop.mp4'
is_portrait = False
scale = 0.5 # Choose any appropiate value to fit your monitor vertically to see the whole video

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total_frames}, FPS (maybe incorrect): {fps}")

frame_idx = 0
start_frame = None
end_frame = None

font_scale = 0.6

# Red color mask (tuned for your ball). Two ranges because of how HSV colour space works
lower_red1 = np.array([0, 120, 120])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 120])
upper_red2 = np.array([180, 255, 255])

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    print("End of video.")
    break

  if not is_portrait:
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

  if scale != 1:
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
  mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
  mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
  mask = cv2.bitwise_or(mask1, mask2)

  if scale != 1:
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

  # Stack original and mask side-by-side
  combined = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))

  # Overlay frame number
  cv2.putText(combined, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)

  if start_frame is not None:
    cv2.putText(combined, f"Start: {start_frame}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
  
  if end_frame is not None:
    cv2.putText(combined, f"End: {end_frame}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)

  cv2.imshow('Video + Mask (Press s=start, e=end, q=quit)', combined)

  key = cv2.waitKey(0) & 0xFF  # Wait for keypress
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

print(f"\nSelected frames → Start: {start_frame}, End: {end_frame}")
