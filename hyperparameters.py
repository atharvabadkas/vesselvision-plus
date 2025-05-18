from typing import Dict, Any, List, Optional, Tuple, Union
import torch

def get_training_hyperparameters(
    device: str = None,
    epochs: int = 200,
    project: str = "vessel_detection",
    name: str = "yolov8s_optimized",
    use_advanced_techniques: bool = True
) -> Dict[str, Any]:

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    hyperparams = {
        'data': 'data.yaml',              # Dataset config
        'epochs': epochs,                 # Number of epochs to train
        'imgsz': 580,                     # Reduced from 640 to 580 for memory efficiency
        'device': device,                 # Device to use
        'patience': 0,                    # Disable early stopping (0 = disabled)
        'batch': 4,                       # Reduced batch size for small dataset and memory constraints
        'optimizer': 'SGD',               # SGD optimizer for better generalization
        'momentum': 0.85,                 # Momentum for SGD optimizer
        'lr0': 0.001,                     # Initial learning rate
        'lrf': 0.12,                      # Final learning rate factor with cosine scheduler
        'weight_decay': 0.0005,           # Increased weight decay for better regularization
        'dropout': 0.3,                   # Increased dropout to prevent overfitting
        'hsv_h': 0.015,                   # HSV hue augmentation
        'hsv_s': 0.7,                     # HSV saturation augmentation
        'hsv_v': 0.4,                     # HSV value augmentation
        'degrees': 20.0,                  # Increased random rotation augmentation
        'translate': 0.2,                 # Random translation augmentation
        'scale': 0.6,                     # Increased random scaling augmentation
        'shear': 0.2,                     # Added shear transformation
        'flipud': 0.1,                    # Added vertical flip
        'fliplr': 0.5,                    # Horizontal flip probability
        'mosaic': 1.0,                    # Enable mosaic augmentation
        'copy_paste': 0.5,                # Copy-paste augmentation
        'save': True,                     # Save checkpoints
        'save_period': -1,                # Save model every X epochs (-1 for best only)
        'plots': True,                    # Generate training plots
        'project': project,               # Project name
        'name': name,                     # Run name
        'exist_ok': True,                 # Overwrite existing run
        'nbs': 64,                        # Nominal batch size for scaling
        'cos_lr': True,                   # Enable cosine LR scheduler
        'overlap_mask': True,             # Improved mask overlap handling
        'val': True,                      # Run validation during training
        'fraction': 1.0,                  # Use all training data
        'rect': True,                     # Enable rectangle training for less padding
        'multi_scale': True,              # Multi-scale training for robustness
        'workers': 4,                     # Reduced data loading workers from 8 to 4
        'max_det': 150,                   # Maximum detections per image (reduced to prevent NMS bottleneck)
        'seed': 42,                       # Random seed for reproducibility
        'verbose': True,                  # Detailed training output
        'resume': False,                  # Don't resume from previous run
        'freeze': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Freeze first 10 layers initially
        'augment': True,                  # Enable augmentation
        'close_mosaic': 0,                # Keep mosaic throughout training
        'amp': True,                      # Enable mixed precision training
        'erasing': 0.5,                   # Increased random erasing probability
        'mixup': 0.5,                     # Increased MixUp augmentation
        'label_smoothing': 0.1,           # Added label smoothing
        'cache': 'ram',                   # Cache images in RAM instead of memory for better efficiency
        'iou': 0.6,                       # NMS IoU threshold (adjusted for vessel detection)
        'conf': 0.25,                     # Confidence threshold (reduced to increase recall)
        'nms': True,                      # Enable NMS
        'agnostic_nms': True,             # Class-agnostic NMS (useful for closely positioned vessels)
    }
    
    if use_advanced_techniques:
        hyperparams.update({
            'epochs': max(epochs, 200),   # Ensure sufficient training time
            'close_mosaic': 0,            # Keep mosaic throughout training
        })
    
    return hyperparams

def get_phase_hyperparameters(
    base_hyperparams: Dict[str, Any],
    phase: int,
    total_epochs: int
) -> Dict[str, Any]:

    phase_params = base_hyperparams.copy()
    
    # Calculate phase epochs
    phase1_end = int(total_epochs * 0.6)  # 60% of total epochs
    phase2_end = int(total_epochs * 0.8)  # 80% of total epochs
    
    if phase == 1:
        # Phase 1: Heavy augmentation (0-60%)
        phase_params.update({
            'epochs': phase1_end,
            'mosaic': 1.0,
            'copy_paste': 0.5,
            'erasing': 0.5,
            'mixup': 0.5,
            'label_smoothing': 0.1,
            'patience': 0,                # Disable early stopping
        })
    elif phase == 2:
        # Phase 2: Medium augmentation (60-80%)
        phase_params.update({
            'epochs': phase2_end - phase1_end,
            'mosaic': 0.0,                # Disable mosaic
            'copy_paste': 0.0,            # Disable copy-paste
            'mixup': 0.3,                 # Keep mixup
            'label_smoothing': 0.1,
            'patience': 0,                # Disable early stopping
        })
    elif phase == 3:
        # Phase 3: Fine-tuning (80-100%)
        phase_params.update({
            'epochs': total_epochs - phase2_end,
            'freeze': list(range(10)),    # Freeze first 10 layers (backbone)
            'lr0': base_hyperparams['lr0'] / 10.0,  # Lower learning rate
            'mosaic': 0.0,
            'copy_paste': 0.0,
            'mixup': 0.0,
            'label_smoothing': 0.05,      # Reduced label smoothing for fine-tuning
            'patience': 0,                # Disable early stopping
        })
    
    # Always remove any potentially invalid parameters 
    invalid_params = ['fl_gamma', 'cutmix', 'random_erasing', 'callbacks']
    for param in invalid_params:
        if param in phase_params:
            del phase_params[param]
    
    return phase_params

def get_device() -> str:

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu" 