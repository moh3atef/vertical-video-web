import cv2
import numpy as np
import os


class VerticalNewsConverter:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
            return x + w // 2

        return frame.shape[1] // 2

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_width = 1080
        output_height = 1920

        crop_width = int(source_height * 9 / 16)
        crop_width = min(crop_width, source_width)

        temp_output = "temp_no_audio.mp4"

        writer = cv2.VideoWriter(
            temp_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height)
        )

        previous_center = source_width // 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            center_x = self.detect_face_center(frame)

            center_x = int(previous_center * 0.85 + center_x * 0.15)
            previous_center = center_x

            x1 = int(center_x - crop_width // 2)
            x1 = max(0, min(x1, source_width - crop_width))

            crop = frame[:, x1:x1 + crop_width]
            resized = cv2.resize(crop, (output_width, output_height))

            writer.write(resized)

        cap.release()
        writer.release()

        os.system(
            f'ffmpeg -y -i "{temp_output}" -i "{input_path}" '
            f'-map 0:v:0 -map 1:a:0 -c:v libx264 -c:a aac -shortest "{output_path}"'
        )

        if os.path.exists(temp_output):
            os.remove(temp_output)
