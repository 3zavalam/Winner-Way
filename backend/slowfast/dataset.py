import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import re
import random

def load_video_opencv(video_path, num_frames=32):
    """
    Load video using OpenCV
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    
    # Sample indices uniformly
    indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i in indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not load any frames from video: {video_path}")
    
    # If we couldn't get enough frames, duplicate the last one
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    # Convert to numpy array
    frames = np.array(frames)
    
    # Convert to torch tensor and normalize
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    
    return frames

class TennisStrokeDataset(Dataset):
    def __init__(self, root_dir, metadata_file=None, transform=None, num_frames=32, 
                 alpha=8, mode='train', use_augmentation=True):
        """
        Tennis stroke dataset with rich metadata support
        
        Args:
            root_dir (string): Directory with all the videos.
            metadata_file (string, optional): Path to the metadata json file.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int): Number of frames to sample from each video.
            alpha (int): Frame rate ratio between slow and fast pathways.
            mode (string): 'train', 'val', or 'test'
            use_augmentation (bool): Whether to use data augmentation
        """
        self.root_dir = root_dir
        self.metadata_file = metadata_file
        self.metadata = {}
        self.use_augmentation = use_augmentation and mode == 'train'
        
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                # Handle both list and dict format for metadata
                data = json.load(f)
                if isinstance(data, list):
                    # Map filenames to metadata for quick lookup
                    self.metadata = {item.get('output_filename', ''): item for item in data}
                else:
                    self.metadata = data
            print(f"Loaded metadata with {len(self.metadata)} entries")
        
        # Find all video files and extract their details
        self.samples = []
        self._find_videos(root_dir)
        
        # Filter samples for train/val/test split (80/10/10 split)
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(self.samples))
        n = len(indices)
        
        if mode == 'train':
            self.samples = [self.samples[i] for i in indices[:int(0.8 * n)]]
        elif mode == 'val':
            self.samples = [self.samples[i] for i in indices[int(0.8 * n):int(0.9 * n)]]
        else:  # test
            self.samples = [self.samples[i] for i in indices[int(0.9 * n):]]
        
        # Override transform with augmentations if needed
        self.base_transform = transform
        self.transform = self._get_transform_with_augmentation() if self.use_augmentation else transform
            
        self.num_frames = num_frames
        self.alpha = alpha
        
        # Create comprehensive class mapping for more detailed classification
        self.stroke_classes = {
            'Forehand_Topspin': 0,
            'Forehand_Slice': 1,
            'Forehand_Flat': 2,
            'Forehand_Drop_Shot': 3,
            'Backhand_Topspin_1H': 4,
            'Backhand_Topspin_2H': 5,
            'Backhand_Slice_1H': 6,
            'Backhand_Flat_1H': 7,
            'Backhand_Flat_2H': 8,
            'Backhand_Drop_Shot': 9,
            'Serve': 10,
            'Smash': 11,
            'Return': 12
        }
        
        # Mapping of dominant hand
        self.hand_mapping = {
            'right': 0,
            'left': 1
        }
        
        # Mapping of camera views
        self.camera_mapping = {
            'behind': 0,
            'side': 1,
            'front': 2,
            'court_level': 3,
            'aerial': 4,
            'unknown': 5
        }
        
        # Print dataset statistics
        self._print_stats()
    
    def _get_transform_with_augmentation(self):
        """
        Get transform pipeline with data augmentation
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),  # Horizontal flip to simulate different court positions
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Simulate different lighting
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Slight affine transforms
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
    
    def _find_videos(self, directory):
        """
        Find all video files in the directory and extract their details
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(root, file)
                    
                    # Try to extract info from the path and filename
                    stroke_info = self._extract_stroke_info(video_path, file)
                    
                    # Check if we have this file in metadata
                    metadata_info = self._get_metadata_for_file(file)
                    
                    # Only add if we have metadata or can derive info from path
                    if metadata_info or stroke_info['stroke_type'] != 'unknown':
                        # Combine path info and metadata
                        sample_info = {**stroke_info, **metadata_info, 'path': video_path}
                        self.samples.append(sample_info)
    
    def _extract_stroke_info(self, video_path, filename):
        """
        Extract stroke information from path and filename
        Format: {player_name}_{stroke_abbr}_{variant_abbr}_{number}.mp4
        """
        # Default values
        info = {
            'player_name': 'unknown',
            'stroke_type': 'unknown',
            'shot_variant': 'unknown',
            'hand_style': 'unknown',
            'dominant_hand': 'right',
            'camera_view': 'unknown',
            'inverted': 'False',
            'surface': 'unknown',
            'filename': filename
        }
        
        # Check if it's a return from path
        if 'return' in video_path.lower():
            info['stroke_type'] = 'Return'
            return info
        
        # Extract basic stroke type from path
        if 'Forehand' in video_path:
            info['stroke_type'] = 'Forehand'
        elif 'Backhand' in video_path:
            info['stroke_type'] = 'Backhand'
        elif 'Serve' in video_path:
            info['stroke_type'] = 'Serve'
            return info
        elif 'Smash' in video_path:
            info['stroke_type'] = 'Smash'
            return info
        
        # Extract handedness for backhand
        if 'Backhand' in video_path:
            if '(1H)' in video_path or '1H' in video_path:
                info['hand_style'] = 'one'
            elif '(2H)' in video_path or '2H' in video_path:
                info['hand_style'] = 'two'
        
        # Extract shot variant from path
        if 'Topspin' in video_path:
            info['shot_variant'] = 'Topspin'
        elif 'Slice' in video_path:
            info['shot_variant'] = 'Slice'
        elif 'Drop_Shot' in video_path:
            info['shot_variant'] = 'Drop_Shot'
        elif 'Flat' in video_path:
            info['shot_variant'] = 'Flat'
        
        # Try to extract from filename if available
        parts = filename.split('_')
        if len(parts) >= 4:
            info['player_name'] = parts[0] + '_' + parts[1]  # Assume first two parts are player name
            
            # Map stroke abbreviation to full name
            stroke_abbr = parts[2]
            if stroke_abbr == 'fh':
                info['stroke_type'] = 'Forehand'
            elif stroke_abbr == 'bh':
                info['stroke_type'] = 'Backhand'
            elif stroke_abbr == 'sv':
                info['stroke_type'] = 'Serve'
            elif stroke_abbr == 'sm':
                info['stroke_type'] = 'Smash'
            
            # Map variant abbreviation to full name
            variant_abbr = parts[3].split('.')[0]  # Remove extension
            if '1' in variant_abbr:
                info['hand_style'] = 'one'
            elif '2' in variant_abbr:
                info['hand_style'] = 'two'
                
            if 'ts' in variant_abbr:
                info['shot_variant'] = 'Topspin'
            elif 'sl' in variant_abbr:
                info['shot_variant'] = 'Slice'
            elif 'ds' in variant_abbr:
                info['shot_variant'] = 'Drop_Shot'
            elif 'fl' in variant_abbr:
                info['shot_variant'] = 'Flat'
            elif 'rt' in variant_abbr:
                info['stroke_type'] = 'Return'
        
        return info
    
    def _get_metadata_for_file(self, filename):
        """
        Get metadata for a file if available
        """
        if not self.metadata:
            return {}
        
        if filename in self.metadata:
            return self.metadata[filename]
            
        for key, item in self.metadata.items():
            if item.get('output_filename') == filename:
                return item
        
        return {}
    
    def _should_flip_horizontally(self, sample):
        """
        Determine if a video should be flipped based on metadata
        to standardize orientation (e.g., convert left-handed to right-handed)
        """
        # Check if this is a left-handed player (and not already inverted)
        dominant_hand = sample.get('dominant_hand', 'right').lower()
        inverted = str(sample.get('inverted', 'False')).lower() == 'true'
        
        # Flip if left-handed player and not already inverted,
        # or right-handed player that is inverted
        return (dominant_hand == 'left' and not inverted) or (dominant_hand == 'right' and inverted)
    
    def _apply_metadata_based_transforms(self, frames, sample):
        """
        Apply transforms based on metadata
        """
        # Convert left-handed shots to right-handed for consistency
        if self._should_flip_horizontally(sample):
            frames = torch.flip(frames, [3])  # Flip horizontally
        
        # Apply additional transforms based on metadata if needed
        # For example, we might apply different processing for different camera views
        
        return frames
    
    def _print_stats(self):
        """
        Print dataset statistics
        """
        stroke_counts = {}
        for sample in self.samples:
            stroke_type = sample.get('stroke_type', 'unknown')
            variant = sample.get('shot_variant', 'unknown')
            hand = sample.get('hand_style', 'unknown') if stroke_type == 'Backhand' else ''
            
            key = f"{stroke_type}"
            if stroke_type in ['Forehand', 'Backhand']:
                key += f"_{variant}"
                if stroke_type == 'Backhand' and hand:
                    key += f"_{hand[0]}H"
            
            stroke_counts[key] = stroke_counts.get(key, 0) + 1
        
        print(f"Dataset statistics ({len(self.samples)} samples):")
        for stroke, count in sorted(stroke_counts.items()):
            print(f"  {stroke}: {count}")
        
        # Also print dominant hand statistics
        hand_counts = {}
        for sample in self.samples:
            hand = sample.get('dominant_hand', 'unknown')
            hand_counts[hand] = hand_counts.get(hand, 0) + 1
        
        print("Dominant hand statistics:")
        for hand, count in sorted(hand_counts.items()):
            print(f"  {hand}: {count}")
        
        # Print camera view statistics
        view_counts = {}
        for sample in self.samples:
            view = sample.get('camera_view', 'unknown')
            view_counts[view] = view_counts.get(view, 0) + 1
        
        print("Camera view statistics:")
        for view, count in sorted(view_counts.items()):
            print(f"  {view}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']
        
        # Load video
        try:
            frames = load_video_opencv(video_path, self.num_frames)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return a dummy tensor in case of error
            frames = torch.zeros((self.num_frames, 3, 224, 224))
        
        # Apply metadata-based transforms
        frames = self._apply_metadata_based_transforms(frames, sample)
        
        # Apply general transformations
        if self.transform:
            frames = self.transform(frames)
        
        # Determine class label based on combined stroke information
        stroke_type = sample.get('stroke_type', 'unknown')
        shot_variant = sample.get('shot_variant', 'unknown')
        hand_style = sample.get('hand_style', 'unknown')
        
        # Build the class key
        class_key = stroke_type
        if stroke_type in ['Forehand', 'Backhand'] and shot_variant != 'unknown':
            if stroke_type == 'Backhand' and hand_style != 'unknown':
                class_key = f"{stroke_type}_{shot_variant}_{hand_style[0]}H"
            else:
                class_key = f"{stroke_type}_{shot_variant}"
        
        # Default to a simpler classification if the specific one is not found
        if class_key not in self.stroke_classes:
            if stroke_type in ['Forehand', 'Backhand']:
                class_key = f"{stroke_type}_Topspin"  # Default to topspin
                if stroke_type == 'Backhand':
                    class_key += "_1H"  # Default to one-handed
        
        # Use a default class if still not found
        label = self.stroke_classes.get(class_key, 0)
        
        # Get additional metadata labels for multi-task learning
        dominant_hand = sample.get('dominant_hand', 'right').lower()
        dominant_hand_label = self.hand_mapping.get(dominant_hand, 0)
        
        camera_view = sample.get('camera_view', 'unknown').lower()
        camera_view_label = self.camera_mapping.get(camera_view, 5)
        
        # For slow pathway, sample frames with larger stride
        slow_indices = np.linspace(0, self.num_frames - 1, self.num_frames // self.alpha, dtype=int)
        slow_frames = frames[slow_indices]
        
        # Fast pathway uses all frames
        fast_frames = frames
        
        return {
            'slow_frames': slow_frames,
            'fast_frames': fast_frames,
            'label': label,
            'dominant_hand': dominant_hand_label,
            'camera_view': camera_view_label,
            'path': video_path,
            'class_name': class_key,
            'metadata': {
                'stroke_type': stroke_type,
                'shot_variant': shot_variant,
                'hand_style': hand_style,
                'dominant_hand': dominant_hand,
                'camera_view': camera_view,
                'inverted': sample.get('inverted', 'False'),
                'surface': sample.get('surface', 'unknown'),
                'slowmo': sample.get('slowmo', 'no')
            }
        }