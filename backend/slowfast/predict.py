import os
import torch
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
from model import SlowFastNetwork
from dataset import load_video_opencv, TennisStrokeDataset
import json
import matplotlib.pyplot as plt

def predict_stroke_type(video_path, model, device, num_frames=32, alpha=8, multi_task=True, metadata=None):
    """
    Predict the stroke type for a given video.
    
    Args:
        video_path (str): Path to the video file.
        model (nn.Module): Trained SlowFast model.
        device (torch.device): Device to run the model on.
        num_frames (int): Number of frames to sample from the video.
        alpha (int): Frame rate ratio between slow and fast pathways.
        multi_task (bool): Whether the model was trained with multi-task learning.
        metadata (dict, optional): Additional metadata for the video.
    
    Returns:
        dict: Prediction results.
    """
    # Set to evaluation mode
    model.eval()
    
    # Load video
    frames = load_video_opencv(video_path, num_frames)
    
    # Apply metadata-based transforms if metadata is provided
    if metadata:
        # Apply flipping for left-handed players if needed
        dominant_hand = metadata.get('dominant_hand', 'right').lower()
        inverted = str(metadata.get('inverted', 'False')).lower() == 'true'
        
        if (dominant_hand == 'left' and not inverted) or (dominant_hand == 'right' and inverted):
            frames = torch.flip(frames, [3])  # Flip horizontally
    
    # Normalize
    transform = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    frames = transform(frames)
    
    # For slow pathway, sample frames with larger stride
    slow_indices = np.linspace(0, num_frames - 1, num_frames // alpha, dtype=int)
    slow_frames = frames[slow_indices]
    
    # Fast pathway uses all frames
    fast_frames = frames
    
    # Add batch dimension
    slow_frames = slow_frames.unsqueeze(0).to(device)
    fast_frames = fast_frames.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        if multi_task:
            stroke_outputs, hand_outputs, camera_outputs = model(slow_frames, fast_frames)
            
            # Get predictions and probabilities
            stroke_probs = torch.nn.functional.softmax(stroke_outputs, dim=1)
            hand_probs = torch.nn.functional.softmax(hand_outputs, dim=1)
            camera_probs = torch.nn.functional.softmax(camera_outputs, dim=1)
            
            stroke_confidence, stroke_preds = torch.max(stroke_probs, 1)
            hand_confidence, hand_preds = torch.max(hand_probs, 1)
            camera_confidence, camera_preds = torch.max(camera_probs, 1)
            
            # Get top-3 stroke predictions
            top3_stroke_probs, top3_stroke_indices = torch.topk(stroke_probs, 3)
            top3_stroke_probs = top3_stroke_probs.squeeze().cpu().numpy()
            top3_stroke_indices = top3_stroke_indices.squeeze().cpu().numpy()
            
            # Mapping from indices to class names
            stroke_classes = {
                0: 'Forehand_Topspin',
                1: 'Forehand_Slice',
                2: 'Forehand_Flat',
                3: 'Forehand_Drop_Shot',
                4: 'Backhand_Topspin_1H',
                5: 'Backhand_Topspin_2H',
                6: 'Backhand_Slice_1H',
                7: 'Backhand_Flat_1H',
                8: 'Backhand_Flat_2H',
                9: 'Backhand_Drop_Shot',
                10: 'Serve',
                11: 'Smash',
                12: 'Return'
            }
            
            hand_classes = ['Right', 'Left']
            camera_classes = ['Behind', 'Side', 'Front', 'Court Level', 'Aerial', 'Unknown']
            
            # Create list of top-3 predictions
            top3_stroke_predictions = []
            for prob, idx in zip(top3_stroke_probs, top3_stroke_indices):
                class_name = stroke_classes.get(idx, 'Unknown')
                top3_stroke_predictions.append((class_name, float(prob)))
            
            result = {
                'stroke': {
                    'prediction': stroke_classes.get(stroke_preds.item(), 'Unknown'),
                    'confidence': float(stroke_confidence.item()),
                    'top3': top3_stroke_predictions
                },
                'hand': {
                    'prediction': hand_classes[hand_preds.item()],
                    'confidence': float(hand_confidence.item())
                },
                'camera': {
                    'prediction': camera_classes[camera_preds.item()],
                    'confidence': float(camera_confidence.item())
                }
            }
        else:
            # Single task model
            outputs = model(slow_frames, fast_frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top3_probs = top3_probs.squeeze().cpu().numpy()
            top3_indices = top3_indices.squeeze().cpu().numpy()
            
            # Mapping from indices to class names
            stroke_classes = {
                0: 'Forehand_Topspin',
                1: 'Forehand_Slice',
                2: 'Forehand_Flat',
                3: 'Forehand_Drop_Shot',
                4: 'Backhand_Topspin_1H',
                5: 'Backhand_Topspin_2H',
                6: 'Backhand_Slice_1H',
                7: 'Backhand_Flat_1H',
                8: 'Backhand_Flat_2H',
                9: 'Backhand_Drop_Shot',
                10: 'Serve',
                11: 'Smash',
                12: 'Return'
            }
            
            # Create list of top-3 predictions
            top3_predictions = []
            for prob, idx in zip(top3_probs, top3_indices):
                class_name = stroke_classes.get(idx, 'Unknown')
                top3_predictions.append((class_name, float(prob)))
            
            result = {
                'stroke': {
                    'prediction': stroke_classes.get(preds.item(), 'Unknown'),
                    'confidence': float(confidence.item()),
                    'top3': top3_predictions
                }
            }
    
    return result

def extract_video_info(video_path):
    """
    Extract information from video path and filename
    """
    filename = os.path.basename(video_path)
    
    # Default values
    info = {
        'player_name': 'unknown',
        'stroke_type': 'unknown',
        'shot_variant': 'unknown',
        'hand_style': 'unknown',
        'dominant_hand': 'right',
        'camera_view': 'unknown',
        'inverted': 'False'
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

def process_video_directory(directory, model, device, num_frames=32, alpha=8, 
                           multi_task=True, metadata_file=None, output_file=None):
    """
    Process all videos in a directory and its subdirectories.
    """
    # Load metadata if provided
    metadata = {}
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                # Map filenames to metadata for quick lookup
                metadata = {item.get('output_filename', ''): item for item in data}
            else:
                metadata = data
        print(f"Loaded metadata with {len(metadata)} entries")
    
    results = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                try:
                    # Extract info from path
                    video_info = extract_video_info(video_path)
                    expected_class = f"{video_info['stroke_type']}"
                    if video_info['stroke_type'] in ['Forehand', 'Backhand'] and video_info['shot_variant'] != 'unknown':
                        expected_class += f"_{video_info['shot_variant']}"
                        if video_info['stroke_type'] == 'Backhand' and video_info['hand_style'] != 'unknown':
                            expected_class += f"_{video_info['hand_style'][0]}H"
                    
                    # Get metadata for this file if available
                    file_metadata = metadata.get(file)
                    
                    # Predict
                    prediction = predict_stroke_type(
                        video_path, model, device, num_frames, alpha, multi_task, file_metadata
                    )
                    
                    # Check if prediction matches expected class
                    predicted_class = prediction['stroke']['prediction']
                    is_correct = predicted_class == expected_class
                    
                    # Add the result
                    result = {
                        'video_path': video_path,
                        'expected_class': expected_class,
                        'prediction': prediction,
                        'is_correct': is_correct
                    }
                    
                    results.append(result)
                    
                    # Print the result
                    print(f"Video: {video_path}")
                    print(f"Expected: {expected_class}")
                    print(f"Predicted: {predicted_class}")
                    print(f"Confidence: {prediction['stroke']['confidence']:.4f}")
                    print(f"Correct: {is_correct}")
                    print(f"Top-3 Predictions: {prediction['stroke']['top3']}")
                    
                    if multi_task:
                        print(f"Hand: {prediction['hand']['prediction']} ({prediction['hand']['confidence']:.4f})")
                        print(f"Camera: {prediction['camera']['prediction']} ({prediction['camera']['confidence']:.4f})")
                    
                    print("-" * 50)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    
    if output_file:
        # Save as JSON for easier analysis
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a CSV version for easy viewing
        csv_file = output_file.replace('.json', '.csv')
        with open(csv_file, 'w') as f:
            f.write("Video Path,Expected Class,Predicted Class,Confidence,Is Correct")
            if multi_task:
                f.write(",Hand,Hand Confidence,Camera,Camera Confidence")
            f.write("\n")
            
            for item in results:
                f.write(f"{item['video_path']},{item['expected_class']},{item['prediction']['stroke']['prediction']},{item['prediction']['stroke']['confidence']:.4f},{item['is_correct']}")
                
                if multi_task:
                    f.write(f",{item['prediction']['hand']['prediction']},{item['prediction']['hand']['confidence']:.4f}")
                    f.write(f",{item['prediction']['camera']['prediction']},{item['prediction']['camera']['confidence']:.4f}")
                
                f.write("\n")
    
    # Print summary statistics
    total = len(results)
    correct = sum(1 for item in results if item['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nSummary Statistics:")
    print(f"Total videos processed: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print accuracy by class
    class_stats = {}
    for item in results:
        expected = item['expected_class']
        if expected not in class_stats:
            class_stats[expected] = {'total': 0, 'correct': 0}
        
        class_stats[expected]['total'] += 1
        if item['is_correct']:
            class_stats[expected]['correct'] += 1
    
    print("\nAccuracy by Class:")
    for cls, stats in sorted(class_stats.items()):
        cls_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cls}: {cls_acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # Create a bar chart of accuracy by class
    if class_stats:
        classes = []
        accuracies = []
        for cls, stats in sorted(class_stats.items()):
            if stats['total'] >= 3:  # Only include classes with at least 3 samples
                classes.append(cls)
                accuracies.append(stats['correct'] / stats['total'])
        
        if classes:
            plt.figure(figsize=(12, 6))
            plt.bar(classes, accuracies)
            plt.title('Accuracy by Class')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_file = output_file.replace('.json', '_accuracy_chart.png') if output_file else 'accuracy_chart.png'
            plt.savefig(chart_file)
            plt.close()
    
    return results

def visualize_predictions(video_path, model, device, output_dir, num_frames=32, alpha=8, 
                         multi_task=True, metadata=None):
    """
    Visualize predictions on selected frames from a video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample frames for visualization
    indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    
    # Get prediction
    prediction = predict_stroke_type(video_path, model, device, num_frames, alpha, multi_task, metadata)
    
    # Setup figure
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            axs[i].imshow(frame_rgb)
            axs[i].set_title(f"Frame {idx}")
            axs[i].axis('off')
    
    cap.release()
    
    # Set overall figure title with prediction info
    if multi_task:
        fig.suptitle(f"Prediction: {prediction['stroke']['prediction']} ({prediction['stroke']['confidence']:.2f})\n"
                    f"Hand: {prediction['hand']['prediction']} ({prediction['hand']['confidence']:.2f}), "
                    f"Camera: {prediction['camera']['prediction']} ({prediction['camera']['confidence']:.2f})",
                    fontsize=14)
    else:
        fig.suptitle(f"Prediction: {prediction['stroke']['prediction']} ({prediction['stroke']['confidence']:.2f})",
                    fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_prediction.png")
    plt.savefig(output_path)
    plt.close()
    
    # Also save a text file with detailed prediction info
    with open(os.path.join(output_dir, f"{os.path.basename(video_path)}_prediction.txt"), 'w') as f:
        f.write(f"Video: {video_path}\n\n")
        f.write(f"Stroke: {prediction['stroke']['prediction']} (confidence: {prediction['stroke']['confidence']:.4f})\n")
        f.write("Top-3 stroke predictions:\n")
        for cls, prob in prediction['stroke']['top3']:
            f.write(f"  {cls}: {prob:.4f}\n")
        
        if multi_task:
            f.write(f"\nHand: {prediction['hand']['prediction']} (confidence: {prediction['hand']['confidence']:.4f})\n")
            f.write(f"Camera: {prediction['camera']['prediction']} (confidence: {prediction['camera']['confidence']:.4f})\n")
    
    print(f"Saved visualization to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Predict tennis strokes using trained SlowFast model')
    parser.add_argument('--video', type=str, help='Path to a single video file')
    parser.add_argument('--directory', type=str, help='Directory containing videos to process')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames to sample')
    parser.add_argument('--alpha', type=int, default=8, help='SlowFast alpha parameter')
    parser.add_argument('--output', type=str, help='Path to save prediction results')
    parser.add_argument('--metadata', type=str, help='Path to metadata file')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    if not args.video and not args.directory:
        parser.error('Either --video or --directory must be provided')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load metadata if provided
    metadata = None
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata, 'r') as f:
            metadata_list = json.load(f)
            metadata = {item.get('output_filename', ''): item for item in metadata_list}
        print(f"Loaded metadata with {len(metadata)} entries")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    # Check if the model is multi-task
    multi_task = checkpoint.get('multi_task', True)
    
    # Define stroke classes
    num_classes = 13
    num_hand_classes = 2
    num_camera_classes = 6
    
    # Create model
    model = SlowFastNetwork(
        num_classes=num_classes,
        num_hand_classes=num_hand_classes,
        num_camera_classes=num_camera_classes,
        alpha=args.alpha,
        multi_task_learning=multi_task
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {args.model_path}")
    print(f"Model uses {'multi-task' if multi_task else 'single-task'} learning")
    
    if args.video:
        # Get filename from path
        filename = os.path.basename(args.video)
        
        # Get metadata for this file if available
        file_metadata = metadata.get(filename) if metadata else None
        
        # Process a single video
        prediction = predict_stroke_type(
            args.video, model, device, args.num_frames, args.alpha, multi_task, file_metadata
        )
        
        # Extract expected class from path
        video_info = extract_video_info(args.video)
        expected_class = f"{video_info['stroke_type']}"
        if video_info['stroke_type'] in ['Forehand', 'Backhand'] and video_info['shot_variant'] != 'unknown':
            expected_class += f"_{video_info['shot_variant']}"
            if video_info['stroke_type'] == 'Backhand' and video_info['hand_style'] != 'unknown':
                expected_class += f"_{video_info['hand_style'][0]}H"
        
        # Print the result
        print(f"Video: {args.video}")
        print(f"Expected class (based on path): {expected_class}")
        print(f"Predicted class: {prediction['stroke']['prediction']}")
        print(f"Confidence: {prediction['stroke']['confidence']:.4f}")
        print(f"Top-3 predictions:")
        for cls, prob in prediction['stroke']['top3']:
            print(f"  {cls}: {prob:.4f}")
        
        if multi_task:
            print(f"Hand: {prediction['hand']['prediction']} ({prediction['hand']['confidence']:.4f})")
            print(f"Camera: {prediction['camera']['prediction']} ({prediction['camera']['confidence']:.4f})")
        
        # Save prediction
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'video_path': args.video,
                    'expected_class': expected_class,
                    'prediction': prediction,
                    'is_correct': expected_class == prediction['stroke']['prediction']
                }, f, indent=2)
        
        # Visualize prediction if requested
        if args.visualize:
            visualize_predictions(
                args.video, model, device, args.vis_dir, args.num_frames, args.alpha, 
                multi_task, file_metadata
            )
    
    elif args.directory:
        # Process all videos in the directory
        process_video_directory(
            args.directory, model, device, args.num_frames, args.alpha, 
            multi_task, args.metadata, args.output
        )

if __name__ == '__main__':
    main()