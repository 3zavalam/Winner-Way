import cv2
import os

def extract_follow_through(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    follow_index = int(total_frames * 0.8)  # 80%

    os.makedirs(output_dir, exist_ok=True)

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == follow_index:
            filename = os.path.join(output_dir, "follow_through.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved follow_through frame to {filename}")
            break

        current_frame += 1

    cap.release()
