import cv2
import numpy as np

class ManualFrameSelection:
    def __init__(self, video_path: str, rotate_video: bool, scale: int, output_video_path:str = None):
        self.video_path = video_path
        self.rotate_video = rotate_video
        self.scale = scale
        self.export_video_path = output_video_path
    
    @staticmethod
    def get_red_colour_mask_values() -> tuple:
        # Red color mask (tuned for your ball). Two ranges because of how HSV colour space works
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])
        return lower_red1, upper_red1, lower_red2, upper_red2
    
    def create_red_mask(self, frame):
        lower_red1, upper_red1, lower_red2, upper_red2 = ManualFrameSelection.get_red_colour_mask_values()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask

    def create_composite_frame(self, frame):
        mask = self.create_red_mask(frame)
        
        original_view = frame.copy()

        # Greyscale mask converted to BGR (for stacking)
        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Mask blended with original frame (in red)
        mask_colored = cv2.merge([mask * 0, mask * 0, mask])  # Blue=0, Green=0, Red=mask
        blended_view = cv2.addWeighted(original_view, 1.0, mask_colored.astype(np.uint8), 0.4, 0)

        # Stack all three views side-by-side
        return np.hstack((original_view, mask_view, blended_view))
    
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

    def overlay_text(self, composite_frame, frame_idx: int, start_frame: int, end_frame: int) -> None:
        text_scale = 0.4
        cv2.putText(composite_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 1)

        if start_frame is not None:
            cv2.putText(composite_frame, f"Start: {start_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 1)
        
        if end_frame is not None:
            cv2.putText(composite_frame, f"End: {end_frame}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 1)

        cv2.imshow('(s=start, e=end, q=quit)', composite_frame)

    def play(self) -> None:
        cap = self.open_file()
        if cap is None:
            return
        self.output_video_information(cap)

        frame_idx = 0
        start_frame = None
        end_frame = None
        output_video = None
        
        while cap.isOpened():
            frame = self.get_frame(cap)
            if frame is None:
                break
            
            composite_frame = self.create_composite_frame(frame)
            
            self.overlay_text(composite_frame, frame_idx, start_frame, end_frame)

            if self.export_video_path is not None and frame_idx == 0:
                height, width = composite_frame.shape[:2]
                fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
                output_video = cv2.VideoWriter(self.export_video_path, fourcc, 30.0, (width, height))

            if output_video:
                output_video.write(composite_frame)
        
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
        if output_video:
            output_video.release() # This has to be done after cap.release()
        cv2.destroyAllWindows()
