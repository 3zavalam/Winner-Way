import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from dataset import TennisStrokeDataset
from model import SlowFastNetwork

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function
    """
    def __init__(self, tasks_weights={'stroke': 1.0, 'hand': 0.2, 'camera': 0.2}):
        super(MultiTaskLoss, self).__init__()
        self.tasks_weights = tasks_weights
        self.stroke_criterion = nn.CrossEntropyLoss()
        self.hand_criterion = nn.CrossEntropyLoss()
        self.camera_criterion = nn.CrossEntropyLoss()
    
    def forward(self, stroke_pred, hand_pred, camera_pred, stroke_true, hand_true, camera_true):
        stroke_loss = self.stroke_criterion(stroke_pred, stroke_true)
        hand_loss = self.hand_criterion(hand_pred, hand_true)
        camera_loss = self.camera_criterion(camera_pred, camera_true)
        
        # Weighted sum of losses
        total_loss = (
            self.tasks_weights['stroke'] * stroke_loss + 
            self.tasks_weights['hand'] * hand_loss + 
            self.tasks_weights['camera'] * camera_loss
        )
        
        return total_loss, stroke_loss, hand_loss, camera_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=25, save_dir='models', multi_task=True):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'train_hand_acc': [], 'val_hand_acc': [],
        'train_camera_acc': [], 'val_camera_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_stroke_loss = 0.0
            running_hand_loss = 0.0
            running_camera_loss = 0.0
            
            y_true_stroke = []
            y_pred_stroke = []
            y_true_hand = []
            y_pred_hand = []
            y_true_camera = []
            y_pred_camera = []
            
            # Iterate over data
            for batch_idx, batch in enumerate(dataloader):
                slow_frames = batch['slow_frames'].to(device)
                fast_frames = batch['fast_frames'].to(device)
                stroke_labels = batch['label'].to(device)
                
                if multi_task:
                    hand_labels = batch['dominant_hand'].to(device)
                    camera_labels = batch['camera_view'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if multi_task:
                        stroke_outputs, hand_outputs, camera_outputs = model(slow_frames, fast_frames)
                        loss, stroke_loss, hand_loss, camera_loss = criterion(
                            stroke_outputs, hand_outputs, camera_outputs,
                            stroke_labels, hand_labels, camera_labels
                        )
                    else:
                        outputs = model(slow_frames, fast_frames)
                        loss = criterion(outputs, stroke_labels)
                        stroke_loss = loss
                        # Dummy values for hand and camera
                        hand_loss = torch.tensor(0.0)
                        camera_loss = torch.tensor(0.0)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * stroke_labels.size(0)
                running_stroke_loss += stroke_loss.item() * stroke_labels.size(0)
                
                if multi_task:
                    running_hand_loss += hand_loss.item() * stroke_labels.size(0)
                    running_camera_loss += camera_loss.item() * stroke_labels.size(0)
                    
                    # Get predictions
                    _, stroke_preds = torch.max(stroke_outputs, 1)
                    _, hand_preds = torch.max(hand_outputs, 1)
                    _, camera_preds = torch.max(camera_outputs, 1)
                    
                    # Collect predictions and targets
                    y_true_stroke.extend(stroke_labels.cpu().numpy())
                    y_pred_stroke.extend(stroke_preds.cpu().numpy())
                    y_true_hand.extend(hand_labels.cpu().numpy())
                    y_pred_hand.extend(hand_preds.cpu().numpy())
                    y_true_camera.extend(camera_labels.cpu().numpy())
                    y_pred_camera.extend(camera_preds.cpu().numpy())
                else:
                    # Get predictions
                    _, stroke_preds = torch.max(outputs, 1)
                    
                    # Collect predictions and targets
                    y_true_stroke.extend(stroke_labels.cpu().numpy())
                    y_pred_stroke.extend(stroke_preds.cpu().numpy())
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f'  {phase} Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_stroke_loss = running_stroke_loss / len(dataloader.dataset)
            epoch_stroke_acc = accuracy_score(y_true_stroke, y_pred_stroke)
            
            if multi_task:
                epoch_hand_loss = running_hand_loss / len(dataloader.dataset)
                epoch_camera_loss = running_camera_loss / len(dataloader.dataset)
                epoch_hand_acc = accuracy_score(y_true_hand, y_pred_hand)
                epoch_camera_acc = accuracy_score(y_true_camera, y_pred_camera)
            else:
                epoch_hand_acc = 0.0
                epoch_camera_acc = 0.0
            
            # Update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_stroke_acc)
                history['train_hand_acc'].append(epoch_hand_acc)
                history['train_camera_acc'].append(epoch_camera_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_stroke_acc)
                history['val_hand_acc'].append(epoch_hand_acc)
                history['val_camera_acc'].append(epoch_camera_acc)
                # Step the learning rate scheduler based on validation loss
                scheduler.step(epoch_loss)
            
            # Print epoch metrics
            print(f'  {phase.capitalize()} Loss: {epoch_loss:.4f}, Stroke Acc: {epoch_stroke_acc:.4f}')
            if multi_task:
                print(f'  {phase.capitalize()} Hand Acc: {epoch_hand_acc:.4f}, Camera Acc: {epoch_camera_acc:.4f}')
            
            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_stroke_acc > best_acc:
                best_acc = epoch_stroke_acc
                best_model_wts = model.state_dict().copy()
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_stroke_acc,
                    'multi_task': multi_task
                }, os.path.join(save_dir, 'best_slowfast_model.pth'))
                print(f'  New best model saved with val accuracy: {epoch_stroke_acc:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_stroke_acc,
                'multi_task': multi_task
            }, os.path.join(save_dir, f'slowfast_checkpoint_epoch_{epoch+1}.pth'))
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training curves
    plot_training_curves(history, save_dir, multi_task)
    
    return model, history

def plot_training_curves(history, save_dir, multi_task=True):
    """
    Plot training curves
    """
    if multi_task:
        # Plot all metrics
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot stroke accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Train Stroke Accuracy')
        plt.plot(history['val_acc'], label='Validation Stroke Accuracy')
        plt.title('Stroke Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot hand accuracy
        plt.subplot(2, 2, 3)
        plt.plot(history['train_hand_acc'], label='Train Hand Accuracy')
        plt.plot(history['val_hand_acc'], label='Validation Hand Accuracy')
        plt.title('Hand Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot camera accuracy
        plt.subplot(2, 2, 4)
        plt.plot(history['train_camera_acc'], label='Train Camera Accuracy')
        plt.plot(history['val_camera_acc'], label='Validation Camera Accuracy')
        plt.title('Camera Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        # Plot just stroke metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))

def evaluate_model(model, test_loader, device, save_dir='models', multi_task=True):
    model.eval()
    
    # Prepare lists to collect predictions and ground truth
    all_stroke_preds = []
    all_stroke_labels = []
    all_hand_preds = []
    all_hand_labels = []
    all_camera_preds = []
    all_camera_labels = []
    all_paths = []
    all_class_names = []
    
    with torch.no_grad():
        for batch in test_loader:
            slow_frames = batch['slow_frames'].to(device)
            fast_frames = batch['fast_frames'].to(device)
            stroke_labels = batch['label'].to(device)
            paths = batch['path']
            class_names = batch['class_name']
            
            if multi_task:
                hand_labels = batch['dominant_hand'].to(device)
                camera_labels = batch['camera_view'].to(device)
                
                # Forward pass
                stroke_outputs, hand_outputs, camera_outputs = model(slow_frames, fast_frames)
                
                # Get predictions
                _, stroke_preds = torch.max(stroke_outputs, 1)
                _, hand_preds = torch.max(hand_outputs, 1)
                _, camera_preds = torch.max(camera_outputs, 1)
                
                # Collect predictions and ground truth
                all_stroke_preds.extend(stroke_preds.cpu().numpy())
                all_stroke_labels.extend(stroke_labels.cpu().numpy())
                all_hand_preds.extend(hand_preds.cpu().numpy())
                all_hand_labels.extend(hand_labels.cpu().numpy())
                all_camera_preds.extend(camera_preds.cpu().numpy())
                all_camera_labels.extend(camera_labels.cpu().numpy())
            else:
                # Forward pass (single task)
                outputs = model(slow_frames, fast_frames)
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions and ground truth
                all_stroke_preds.extend(preds.cpu().numpy())
                all_stroke_labels.extend(stroke_labels.cpu().numpy())
            
            all_paths.extend(paths)
            all_class_names.extend(class_names)
    
    # Calculate accuracy
    stroke_accuracy = accuracy_score(all_stroke_labels, all_stroke_preds)
    print(f'Stroke Classification Accuracy: {stroke_accuracy:.4f}')
    
    if multi_task:
        hand_accuracy = accuracy_score(all_hand_labels, all_hand_preds)
        camera_accuracy = accuracy_score(all_camera_labels, all_camera_preds)
        print(f'Hand Classification Accuracy: {hand_accuracy:.4f}')
        print(f'Camera Classification Accuracy: {camera_accuracy:.4f}')
    
    # Get unique class names
    unique_class_names = sorted(list(set(all_class_names)))
    
    # Plot confusion matrix for stroke classification
    cm = confusion_matrix(all_stroke_labels, all_stroke_preds)
    
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Stroke Classification Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_class_names))
    plt.xticks(tick_marks, unique_class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, unique_class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'stroke_confusion_matrix.png'))
    
    # Print classification report
    stroke_report = classification_report(all_stroke_labels, all_stroke_preds, target_names=unique_class_names)
    print("\nStroke Classification Report:")
    print(stroke_report)
    
    # Save report to file
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f'Stroke Classification Accuracy: {stroke_accuracy:.4f}\n\n')
        f.write(stroke_report + '\n\n')
        
        if multi_task:
            f.write(f'Hand Classification Accuracy: {hand_accuracy:.4f}\n')
            f.write(f'Camera Classification Accuracy: {camera_accuracy:.4f}\n')
            
            # Get hand class names
            hand_classes = ['Right', 'Left']
            hand_report = classification_report(all_hand_labels, all_hand_preds, target_names=hand_classes)
            f.write("\nHand Classification Report:\n")
            f.write(hand_report + '\n\n')
            
            # Get camera class names
            camera_classes = ['Behind', 'Side', 'Front', 'Court Level', 'Aerial', 'Unknown']
            camera_report = classification_report(all_camera_labels, all_camera_preds, target_names=camera_classes)
            f.write("\nCamera Classification Report:\n")
            f.write(camera_report)
    
    # Save misclassified examples
    misclassified = [(path, unique_class_names[pred], unique_class_names[label]) 
                     for path, pred, label in zip(all_paths, all_stroke_preds, all_stroke_labels) 
                     if pred != label]
    
    if misclassified:
        with open(os.path.join(save_dir, 'misclassified.txt'), 'w') as f:
            f.write("Path,Predicted,Actual\n")
            for path, pred, label in misclassified:
                f.write(f"{path},{pred},{label}\n")
    
    # If multi-task, plot confusion matrices for hand and camera as well
    if multi_task:
        # Hand confusion matrix
        hand_classes = ['Right', 'Left']
        hand_cm = confusion_matrix(all_hand_labels, all_hand_preds)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(hand_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Hand Classification Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(hand_classes))
        plt.xticks(tick_marks, hand_classes)
        plt.yticks(tick_marks, hand_classes)
        
        # Add text annotations
        thresh = hand_cm.max() / 2.
        for i in range(hand_cm.shape[0]):
            for j in range(hand_cm.shape[1]):
                plt.text(j, i, format(hand_cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if hand_cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'hand_confusion_matrix.png'))
        
        # Camera confusion matrix
        camera_classes = ['Behind', 'Side', 'Front', 'Court Level', 'Aerial', 'Unknown']
        camera_cm = confusion_matrix(all_camera_labels, all_camera_preds)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(camera_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Camera View Classification Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(camera_classes))
        plt.xticks(tick_marks, camera_classes)
        plt.yticks(tick_marks, camera_classes)
        
        # Add text annotations
        thresh = camera_cm.max() / 2.
        for i in range(camera_cm.shape[0]):
            for j in range(camera_cm.shape[1]):
                plt.text(j, i, format(camera_cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if camera_cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'camera_confusion_matrix.png'))
    
    return stroke_accuracy

def main():
    parser = argparse.ArgumentParser(description='Train SlowFast network for tennis stroke classification')
    parser.add_argument('--data_dir', type=str, default='data/videos', help='Directory with all the videos')
    parser.add_argument('--metadata', type=str, default=None, help='Path to the metadata json file (optional)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames to sample from each video')
    parser.add_argument('--alpha', type=int, default=8, help='SlowFast alpha parameter')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models and results')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on the test set')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for evaluation')
    parser.add_argument('--single_task', action='store_true', help='Use single task learning (stroke only)')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations (for non-augmented data)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    
    # Create datasets
    train_dataset = TennisStrokeDataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
        transform=transform,
        num_frames=args.num_frames,
        alpha=args.alpha,
        mode='train',
        use_augmentation=not args.no_augmentation
    )
    
    val_dataset = TennisStrokeDataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
        transform=transform,
        num_frames=args.num_frames,
        alpha=args.alpha,
        mode='val',
        use_augmentation=False  # No augmentation for validation
    )
    
    test_dataset = TennisStrokeDataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
        transform=transform,
        num_frames=args.num_frames,
        alpha=args.alpha,
        mode='test',
        use_augmentation=False  # No augmentation for test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Get the number of classes from the dataset
    num_classes = len(train_dataset.stroke_classes)
    num_hand_classes = len(train_dataset.hand_mapping)
    num_camera_classes = len(train_dataset.camera_mapping)
    
    # Whether to use multi-task learning
    multi_task = not args.single_task
    
    # Create model
    model = SlowFastNetwork(
        num_classes=num_classes,
        num_hand_classes=num_hand_classes,
        num_camera_classes=num_camera_classes,
        alpha=args.alpha,
        multi_task_learning=multi_task
    ).to(device)
    
    # If evaluating, load pretrained model
    if args.evaluate and args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Check if the saved model is multi-task
        saved_multi_task = checkpoint.get('multi_task', True)
        print(f"Loaded model from {args.model_path}")
        print(f"Model uses {'multi-task' if saved_multi_task else 'single-task'} learning")
        evaluate_model(model, test_loader, device, args.save_dir, saved_multi_task)
        return
    
    # Loss function and optimizer
    if multi_task:
        criterion = MultiTaskLoss(tasks_weights={'stroke': 1.0, 'hand': 0.2, 'camera': 0.2})
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    print(f"Using {'multi-task' if multi_task else 'single-task'} learning")
    print(f"Using {'augmented' if not args.no_augmentation else 'non-augmented'} training data")
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        multi_task=multi_task
    )
    
    # Evaluate the model
    evaluate_model(model, test_loader, device, args.save_dir, multi_task)

if __name__ == '__main__':
    main()