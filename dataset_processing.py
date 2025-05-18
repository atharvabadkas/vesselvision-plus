import os
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

def load_dataset_config(data_yaml_path: str = 'data.yaml') -> Dict[str, Any]:

    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config
    except Exception as e:
        raise RuntimeError(f"Error loading data configuration: {e}")

def verify_dataset_structure(data_yaml_path: str = 'data.yaml') -> Tuple[bool, Dict[str, Any]]:

    try:
        # Load dataset configuration
        print(f"Loading dataset config from: {data_yaml_path}")
        data_config = load_dataset_config(data_yaml_path)
        
        # Get base directory of data.yaml for resolving relative paths
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        
        # Get paths and resolve them relative to the data.yaml file location
        train_path = Path(os.path.join(yaml_dir, data_config['train']))
        val_path = Path(os.path.join(yaml_dir, data_config['val']))
        classes = data_config['names']
        
        print(f"Train path (absolute): {train_path}")
        print(f"Val path (absolute): {val_path}")
        
        # Get labels paths from config or use default structure
        if 'train_labels' in data_config and 'val_labels' in data_config:
            train_labels_path = Path(os.path.join(yaml_dir, data_config['train_labels']))
            val_labels_path = Path(os.path.join(yaml_dir, data_config['val_labels']))
            print(f"Using label paths from config: {train_labels_path}, {val_labels_path}")
        else:
            # Default YOLOv8 structure (labels are in parent/labels/images_dir_name)
            train_labels_path = train_path.parent / 'labels' / train_path.name
            val_labels_path = val_path.parent / 'labels' / val_path.name
            print(f"Using default label paths: {train_labels_path}, {val_labels_path}")
        
        stats = {
            "num_classes": len(classes),
            "classes": list(classes.values()),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_labels_path": str(train_labels_path),
            "val_labels_path": str(val_labels_path),
        }
        
        # Check if directories exist
        if not train_path.exists():
            stats["error"] = f"Training directory not found at {train_path}"
            return False, stats
        
        if not val_path.exists():
            stats["error"] = f"Validation directory not found at {val_path}"
            return False, stats
            
        if not train_labels_path.exists():
            stats["error"] = f"Training labels directory not found at {train_labels_path}"
            return False, stats
            
        if not val_labels_path.exists():
            stats["error"] = f"Validation labels directory not found at {val_labels_path}"
            return False, stats
        
        # Count images
        train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
        val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
        
        stats["train_images"] = len(train_images)
        stats["val_images"] = len(val_images)
        stats["total_images"] = len(train_images) + len(val_images)
        
        # Check for label files
        train_labels = list(train_labels_path.glob('*.txt'))
        val_labels = list(val_labels_path.glob('*.txt'))
        
        stats["train_labels"] = len(train_labels)
        stats["val_labels"] = len(val_labels)
        
        # Check if all images have corresponding label files
        train_missing = [img.stem for img in train_images if not (train_labels_path / f"{img.stem}.txt").exists()]
        val_missing = [img.stem for img in val_images if not (val_labels_path / f"{img.stem}.txt").exists()]
        
        stats["train_missing_labels"] = train_missing[:5] if train_missing else []
        stats["val_missing_labels"] = val_missing[:5] if val_missing else []
        stats["train_missing_count"] = len(train_missing)
        stats["val_missing_count"] = len(val_missing)
        
        # Success criteria
        success = (len(train_images) > 0 and 
                  len(val_images) > 0 and 
                  len(train_labels) > 0 and
                  len(val_labels) > 0)
        
        return success, stats
        
    except Exception as e:
        return False, {"error": str(e)}

def generate_class_weights(class_stats: Dict, normalize: bool = True) -> Dict[str, float]:

    # Calculate inverse frequency weighting 
    total_detections = sum(stats["count"] for stats in class_stats.values())
    
    class_weights = {}
    for class_name, stats in class_stats.items():
        # Inverse frequency (rarer classes get higher weights)
        inv_freq = total_detections / max(1, stats["count"])
        
        # Inverse confidence (less confident classes get higher weights)
        inv_conf = 1.0 / max(0.1, stats["avg_confidence"])
        
        # Combined weight (normalized later)
        class_weights[class_name] = inv_freq * inv_conf
    
    # Normalize weights so average is 1.0
    if normalize and class_weights:
        avg_weight = sum(class_weights.values()) / len(class_weights)
        normalized_weights = {cls: w / avg_weight for cls, w in class_weights.items()}
        return normalized_weights
    
    return class_weights

def load_class_weights(weights_path: str) -> Optional[Dict[str, float]]:

    import json
    
    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        return weights
    except Exception as e:
        print(f"Error loading class weights: {e}")
        return None

def apply_test_time_augmentation(model, image, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Apply TTA with the model's built-in augment parameter
    results = model.predict(
        image, 
        device=device, 
        augment=True,  # This enables TTA in YOLOv8
        verbose=False
    )
    
    return results 