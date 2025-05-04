import os
import cv2
import argparse
import numpy as np
from dataset import load_video_opencv
import matplotlib.pyplot as plt

def extract_frames(video_path, output_dir, num_frames=8):
    """
    Extract frames from a video and save them to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        print(f"No frames found in video: {video_path}")
        return
    
    # Sample indices uniformly
    indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    # Extract and save frames
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i in indices:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Save frame
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved {output_path}")
    
    cap.release()
    
    # Create a montage of frames
    if frames:
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        axs = axs.flatten()
        
        for i, frame in enumerate(frames):
            if i < len(axs):
                axs[i].imshow(frame)
                axs[i].set_title(f"Frame {indices[i]}")
                axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "montage.jpg"))
        plt.close()
        print(f"Saved montage to {os.path.join(output_dir, 'montage.jpg')}")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('--video', type=str, required=True, help='Path to a video file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save frames')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to extract')
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output_dir, args.num_frames)

if __name__ == '__main__':
    main()