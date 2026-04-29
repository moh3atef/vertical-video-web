import cv2
import numpy as np
import subprocess
import os

class VerticalNewsConverter:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_face_center(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            center_x = x + w // 2
        else:
            center_x = frame.shape[1] // 2

        return center_x

    def convert_to_vertical(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_width = int(height * 9 / 16)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, height))

        prev_center = width // 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            center_x = self.detect_face_center(frame)

            # smoothing
            center_x = int(prev_center * 0.8 + center_x * 0.2)
            prev_center = center_x

            x1 = max(0, center_x - target_width // 2)
            x2 = x1 + target_width

            if x2 > width:
                x2 = width
                x1 = width - target_width

            crop = frame[:, x1:x2]

            out.write(crop)

        cap.release()
        out.release()

    def process_video(self, input_path):
        filename = os.path.basename(input_path)
        output_path = f"output_{filename}"

        self.convert_to_vertical(input_path, output_path)

        return output_path
