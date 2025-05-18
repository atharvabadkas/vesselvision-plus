import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path

# Add the parent directory to sys.path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from YOLOv8.train import train_yolo
from modules.modules_initalization import get_device

# Global variables for handling pause/resume
CHECKPOINT_FILE = 'training_checkpoint.json'
interrupted = False

def format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def signal_handler(sig, frame):
    global interrupted
    print("\n\nTraining pause requested. Saving checkpoint and exiting...")
    interrupted = True

def save_checkpoint(current_epoch, total_epochs, batch, lr, name, model_path, data_yaml, use_advanced, use_phased):
    checkpoint = {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'batch': batch,
        'lr': lr,
        'name': name,
        'model_path': model_path,
        'data_yaml': data_yaml,
        'use_advanced': use_advanced,
        'use_phased': use_phased,
        'timestamp': time.time()
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=4)
    
    print(f"Checkpoint saved: {CHECKPOINT_FILE}")
    print(f"Resume training later with: python YOLOv8/run.py --resume")

def load_checkpoint():
    """Load training state from checkpoint file"""
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"No checkpoint file found: {CHECKPOINT_FILE}")
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        
        print("\nFound training checkpoint:")
        print(f"- Name: {checkpoint['name']}")
        print(f"- Progress: {checkpoint['current_epoch']}/{checkpoint['total_epochs']} epochs")
        print(f"- Saved: {format_time(time.time() - checkpoint['timestamp'])} ago")
        
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="YOLO Training with Pause/Resume Support")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "=" * 80)
    print("VESSEL DETECTION TRAINING")
    print("=" * 80)
    
    # Default training parameters
    epochs = 100
    batch = 4
    lr = 0.001
    name = "yolov8s_improved"
    use_advanced = False
    use_phased = True
    model_path = 'yolov8s.pt'  # Use pretrained YOLOv8s instead of looking for existing weights
    current_epoch = 0  # Starting epoch
    
    # Use the current working directory to create an absolute path to data.yaml
    current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_yaml = os.path.join(current_dir, 'data.yaml')
    
    # Check if resuming from checkpoint
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            current_epoch = checkpoint['current_epoch']
            epochs = checkpoint['total_epochs']
            batch = checkpoint['batch']
            lr = checkpoint['lr']
            name = checkpoint['name']
            use_advanced = checkpoint['use_advanced']
            use_phased = checkpoint['use_phased']
            model_path = checkpoint['model_path']
            data_yaml = checkpoint['data_yaml']
            
            # Calculate remaining epochs
            remaining_epochs = epochs - current_epoch
            
            print(f"\nResuming training from epoch {current_epoch}")
            print(f"Remaining epochs: {remaining_epochs}")
        else:
            print("No valid checkpoint found. Starting new training session.")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: data.yaml not found. Please place it in the project root directory: {current_dir}")
        return False
    
    print("\nTraining Configuration:")
    print(f"- Epochs: {epochs}" + (f" (resuming from {current_epoch})" if current_epoch > 0 else ""))
    print(f"- Batch size: {batch}")
    print(f"- Learning rate: {lr}")
    print(f"- Run name: {name}")
    print(f"- Using dataset config: {data_yaml}")
    print(f"- Press Ctrl+C to pause training at any time")
    
    start_time = time.time()
    # Add the dataset limitations and advanced regularization techniques
    custom_params = {
        'batch': batch,
        'lr0': lr,
        'data': data_yaml,
        'imgsz': 640,
        'patience': 10,
        'cache': True,
        'rect': True,
        'weight_decay': 0.0005,
        'dropout': 0.3,
        'close_mosaic': 0,
        'degrees': 20.0,
        'scale': 0.6,
        'shear': 0.2,
        'flipud': 0.1,
        'mixup': 0.5,
        'erasing': 0.5,
        'label_smoothing': 0.1,
        'freeze': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    
    # If resuming, find the best weights file to resume from
    if current_epoch > 0:
        project_dir = os.path.join("runs", "detect", name)
        weights_dir = os.path.join(project_dir, "weights")
        
        if os.path.exists(weights_dir):
            # Look for the last checkpoint weights
            weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
            if weight_files:
                # Either find epoch-specific weight or use best.pt
                epoch_weights = [f for f in weight_files if f.startswith(f"epoch{current_epoch}")]
                if epoch_weights:
                    model_path = os.path.join(weights_dir, epoch_weights[0])
                elif "best.pt" in weight_files:
                    model_path = os.path.join(weights_dir, "best.pt")
                else:
                    # Sort by modified time and get the most recent
                    weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(weights_dir, x)), reverse=True)
                    model_path = os.path.join(weights_dir, weight_files[0])
                
                print(f"Resuming from weights: {model_path}")
    
    try:
        print("\nStarting training...")
        
        # Update parameters for resuming
        if current_epoch > 0:
            custom_params['resume'] = True
            remaining_epochs = epochs - current_epoch
            
            # When resuming, we need to specify exactly how many epochs to train
            results = train_yolo(
                epochs=remaining_epochs,  # Only train the remaining epochs
                name=name,
                model_path=model_path,
                custom_hyperparams=custom_params,
                use_advanced_techniques=use_advanced,
                use_phased_training=use_phased,
                current_epoch=current_epoch
            )
        else:
            # Normal training from the beginning
            results = train_yolo(
                epochs=epochs,
                name=name,
                model_path=model_path,
                custom_hyperparams=custom_params,
                use_advanced_techniques=use_advanced,
                use_phased_training=use_phased
            )
        
        duration = time.time() - start_time
        formatted_time = format_time(duration)
        
        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETED in {formatted_time}")
        print(f"Results saved in: {results.save_dir}")
        print("=" * 80)
        
        # Remove checkpoint file once completed successfully
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        
        return results
        
    except KeyboardInterrupt:
        # If keyboard interrupt (Ctrl+C) is detected, save checkpoint
        end_epoch = current_epoch + int((time.time() - start_time) / 3600 * batch / epochs)  # Estimate current epoch
        save_checkpoint(end_epoch, epochs, batch, lr, name, model_path, data_yaml, use_advanced, use_phased)
        print("\nTraining paused. You can resume later.")
        return False
    except Exception as e:
        print(f"\nError during training: {e}")
        # Save checkpoint on error as well
        end_epoch = current_epoch + int((time.time() - start_time) / 3600 * batch / epochs)  # Estimate current epoch
        save_checkpoint(end_epoch, epochs, batch, lr, name, model_path, data_yaml, use_advanced, use_phased)
        print("Training state saved. You can resume after fixing the error.")
        return False

if __name__ == "__main__":
    main() 