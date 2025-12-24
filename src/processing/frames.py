import cv2
import base64
import os

class FrameExtractor:
    def __init__(self, resize_width: int = 640):
        self.resize_width = resize_width
    
    def extract(self, video_path: str) -> list[str]:
        frames = []
        frame_limit = 35

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise FileNotFoundError(f"Video file is not found at {video_path}")

        try:
            fps = vid.get(cv2.CAP_PROP_FPS)
            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # has to be an int for range() to work
            duration = frame_count / fps
            frame_interval = duration / frame_limit
            interval = int(frame_interval * fps)

            for i in range(0, frame_count, interval):
                vid.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = vid.read()
                if not success:
                    raise RuntimeError(f"Failed to read frame {i}")
                frame = cv2.resize(frame, (self.resize_width, int(frame.shape[0] * self.resize_width / frame.shape[1])))
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
        finally:
            vid.release()

        return frames