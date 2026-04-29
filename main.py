import cv2
import numpy as np
import mediapipe as mp
import ffmpeg
import os


class VerticalNewsConverter:
    def __init__(self):
        self.output_width = 1080
        self.output_height = 1920

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        )

        self.prev_gray = None

    def is_cut(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True

        diff = cv2.absdiff(gray, self.prev_gray)
        score = np.mean(diff)

        self.prev_gray = gray

        return score > 22

    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)

        h, w, _ = frame.shape
        faces = []

        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                faces.append([x, y, bw, bh])

        return faces

    def detect_graphics_top(self, frame):
        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lower = gray[int(h * 0.45):h, :]

        edges = cv2.Canny(lower, 80, 160)
        density = edges.mean(axis=1)

        rows = np.where(density > 25)[0]

        if len(rows) == 0:
            return h

        y = int(h * 0.45) + int(rows[0])

        if y < int(h * 0.52):
            return h

        return y

    def analyze_scene(self, frame):
        faces = self.detect_faces(frame)

        h, w, _ = frame.shape

        left = 0
        right = 0

        for f in faces:
            cx = f[0] + f[2] // 2
            if cx < w // 2:
                left += 1
            else:
                right += 1

        is_split = (left > 0 and right > 0)

        if faces:
            largest = max(faces, key=lambda b: b[2] * b[3])
            center_x = largest[0] + largest[2] // 2
        else:
            center_x = None

        graphics_y = self.detect_graphics_top(frame)

        return {
            "is_split": is_split,
            "center_x": center_x,
            "graphics_y": graphics_y
        }

    def crop_single(self, frame, center_x, graphics_y):
        h, w, _ = frame.shape

        if graphics_y is None:
            graphics_y = h

        # Safety margin above graphics
        clean_bottom = graphics_y - int(h * 0.04)

        if clean_bottom < int(h * 0.6):
            clean_bottom = h

        clean = frame[0:clean_bottom, :]
        ch, cw, _ = clean.shape

        crop_w = int(ch * 9 / 16)

        if crop_w > cw:
            crop_w = cw

        if center_x is None:
            center_x = cw // 2

        x1 = int(center_x - crop_w // 2)
        x1 = max(0, min(x1, cw - crop_w))

        crop = clean[:, x1:x1 + crop_w]

        return cv2.resize(crop, (self.output_width, self.output_height))

    def crop_split_half(self, half_frame, graphics_y):
        h, w, _ = half_frame.shape

        if graphics_y is None:
            graphics_y = h

        # Safety margin above graphics
        clean_bottom = graphics_y - int(h * 0.04)

        if clean_bottom < int(h * 0.6):
            clean_bottom = h

        clean = half_frame[0:clean_bottom, :]
        ch, cw, _ = clean.shape

        target_w = self.output_width
        target_h = self.output_height // 2

        scale = target_h / ch

        new_w = int(cw * scale)
        new_h = target_h

        resized = cv2.resize(clean, (new_w, new_h))

        if new_w < target_w:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            canvas[:, x_offset:x_offset + new_w] = resized
            return canvas

        x1 = (new_w - target_w) // 2
        return resized[:, x1:x1 + target_w]

    def crop_split(self, frame, graphics_y):
        h, w, _ = frame.shape

        left = frame[:, :w // 2]
        right = frame[:, w // 2:]

        top = self.crop_split_half(right, graphics_y)
        bottom = self.crop_split_half(left, graphics_y)

        return np.vstack([top, bottom])

    def process_video(self, input_path, output_path):
        temp = "temp_v7_1.mp4"

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            temp,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.output_width, self.output_height)
        )

        current_decision = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.is_cut(frame):
                current_decision = self.analyze_scene(frame)

            if current_decision["is_split"]:
                out = self.crop_split(frame, current_decision["graphics_y"])
            else:
                out = self.crop_single(
                    frame,
                    current_decision["center_x"],
                    current_decision["graphics_y"]
                )

            writer.write(out)

        cap.release()
        writer.release()

        self.attach_audio(input_path, temp, output_path)

        if os.path.exists(temp):
            os.remove(temp)

    def attach_audio(self, original, processed, out):
        video = ffmpeg.input(processed)
        audio = ffmpeg.input(original).audio

        (
            ffmpeg
            .output(
                video.video,
                audio,
                out,
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p"
            )
            .overwrite_output()
            .run()
        )


if __name__ == "__main__":
    converter = VerticalNewsConverter()
    converter.process_video("input.mp4", "output_v7_1.mp4")
