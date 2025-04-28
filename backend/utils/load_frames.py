import cv2

def load_video_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return frames