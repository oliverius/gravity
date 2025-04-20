import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# === CONFIGURATION ===
video_path = "videos/ball drop.mp4"
floor_to_ceiling_distance = 232 # cm
ball_diameter = 4 # cm
real_distance_m = (floor_to_ceiling_distance - ball_diameter) / 100
is_portrait = False
start_frame = 118
end_frame = 201

# === SETUP ===
cap = cv2.VideoCapture(video_path)
fps = 120 # Recorded in slow motion but the video will show 29.994986fps cap.get(cv2.CAP_PROP_FPS)
print(f"Detected FPS: {fps} (verify this is correct for your video!)")

positions_px = []
times = []

frame_idx = 0
relative_idx = 0  # Used for timing

# Red color mask (tuned for your ball). Two ranges because of how HSV colour space works
lower_red1 = np.array([0, 120, 120])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 120])
upper_red2 = np.array([180, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    if frame_idx < start_frame:
      frame_idx += 1
      continue
    if frame_idx > end_frame:
      break

    if not is_portrait:
      frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # === RED BALL DETECTION (HSV) ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
      largest = max(contours, key=cv2.contourArea)
      if cv2.contourArea(largest) > 50:
        M = cv2.moments(largest)
        if M['m00'] != 0:
            cy = int(M['m01'] / M['m00'])  # Vertical center
            positions_px.append(cy)
            times.append(relative_idx / fps)

    frame_idx += 1
    relative_idx += 1

cap.release()

# === VALIDATION ===
if len(positions_px) < 5:
  print("Not enough data points. Try checking your frame range or color mask.")
  exit()

# === PIXEL TO METER CONVERSION ===
start_px = min(positions_px)
end_px = max(positions_px)
print(positions_px)
total_px = end_px - start_px

camera_tilt_deg = 5
correction_factor = 1 / math.cos(math.radians(camera_tilt_deg))
real_distance_m = real_distance_m * correction_factor
pixel_to_meter = real_distance_m / total_px


positions_m = [(p - start_px) * pixel_to_meter for p in positions_px]

# === FIT y(t) = a*t² + b*t + c
coeffs = np.polyfit(times, positions_m, 2)
a = coeffs[0]
g_estimate = 2 * a

print(f"\nEstimated gravitational acceleration: {g_estimate:.4f} m/s²")

# === PLOT RESULTS ===
t = np.array(times)
y = np.array(positions_m)
fit_y = np.polyval(coeffs, t)

plt.figure()
plt.plot(t, y, 'bo', label='Measured')
plt.plot(t, fit_y, 'r-', label='Fitted parabola')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Position (m)')
plt.title('Ball Drop Trajectory and Gravity Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
