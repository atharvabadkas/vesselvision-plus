import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import Counter
import time

# Import necessary PyTorch modules
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import custom utilities using modules_initalization
from YOLOv8.model_enhancement import create_enhanced_model, create_training_callbacks, create_safe_transition_callback
from YOLOv8.modules.modules_initalization import (
    get_training_hyperparameters,
    get_phase_hyperparameters,
    get_device,
    load_class_weights,
    plot_class_weights
)

def train_with_phases(
    model: YOLO, 
    hyperparams: Dict[str, Any],
    epochs: int = 200,
    use_advanced_techniques: bool = False,  # Temporarily disable for debugging
) -> Any:
    # Calculate phase epochs
    phase1_end = int(epochs * 0.6)  # 60% of total epochs
    phase2_end = int(epochs * 0.8)  # 80% of total epochs
    
    print("\n" + "=" * 50)
    print(f"Phase 1: Heavy Augmentation (epochs 0-{phase1_end})")
    print("=" * 50)
    
    if use_advanced_techniques:
        original_model = model
        model = create_enhanced_model(
            model_path=model.ckpt_path,
            use_coordattn=True,
            use_contrastive_loss=True
        )
    
    # Get phase 1 hyperparameters
    phase1_params = get_phase_hyperparameters(hyperparams, phase=1, total_epochs=epochs)
    
    # Ensure we don't include invalid parameters
    for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
        if param in phase1_params:
            del phase1_params[param]
    
    # Disable early stopping for proper phases
    phase1_params['patience'] = 0
    
    # Train phase 1
    results1 = model.train(**phase1_params)
    
    # Phase 2: Medium augmentation
    print("\n" + "=" * 50)
    print(f"Phase 2: Medium Augmentation (epochs {phase1_end+1}-{phase2_end})")
    print("=" * 50)
    
    # Use the last checkpoint from phase 1 instead of best model
    last_checkpoint = str(Path(results1.save_dir) / 'weights' / 'last.pt')
    model = YOLO(last_checkpoint)
    
    if use_advanced_techniques:
        model = create_enhanced_model(
            model_path=last_checkpoint,
            use_coordattn=True,
            use_contrastive_loss=True
        )
    
    # Get phase 2 hyperparameters
    phase2_params = get_phase_hyperparameters(hyperparams, phase=2, total_epochs=epochs)
    
    # Ensure we don't include invalid parameters
    for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
        if param in phase2_params:
            del phase2_params[param]
    
    # Disable early stopping for proper phases
    phase2_params['patience'] = 0
    
    # Train phase 2
    results2 = model.train(**phase2_params)
    
    # Phase 3: Backbone freeze and fine-tuning
    print("\n" + "=" * 50)
    print(f"Phase 3: Freezing Backbone (epochs {phase2_end+1}-{epochs})")
    print("=" * 50)
    
    # Use the last checkpoint from phase 2
    last_checkpoint2 = str(Path(results2.save_dir) / 'weights' / 'last.pt')
    model = YOLO(last_checkpoint2)
    
    # Get phase 3 hyperparameters
    phase3_params = get_phase_hyperparameters(hyperparams, phase=3, total_epochs=epochs)
    
    # Ensure we don't include invalid parameters
    for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
        if param in phase3_params:
            del phase3_params[param]
    
    # Disable early stopping for proper phases
    phase3_params['patience'] = 0
    
    # Train phase 3
    results3 = model.train(**phase3_params)
    
    return results3

def train_yolo(
    epochs: int = 200,
    project: str = "vessel_detection",
    name: str = "yolov8s_optimized",
    use_wandb: bool = False,
    model_path: str = 'yolov8s.pt',
    custom_hyperparams: Dict = None,
    use_advanced_techniques: bool = False,  # Temporarily disable for debugging
    use_phased_training: bool = True,
    generate_advanced_statistics: bool = True,
    current_epoch: int = 0  # Added parameter for resuming from a specific epoch
) -> Any:
    
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "=" * 50)
    if use_advanced_techniques:
        print("TRAINING YOLOv8s MODEL WITH ADVANCED TECHNIQUES")
    else:
        print("TRAINING YOLOv8s MODEL (STANDARD MODE)")
    print("=" * 50)
    
    # Get hyperparameters
    hyperparams = get_training_hyperparameters(
        device=device,
        epochs=epochs,
        project=project,
        name=name,
        use_advanced_techniques=use_advanced_techniques
    )
    
    # Override with custom hyperparameters if provided
    if custom_hyperparams:
        hyperparams.update(custom_hyperparams)
    
    # Add resume flag if resuming from a specific epoch
    if current_epoch > 0:
        print(f"Resuming training from epoch {current_epoch}")
        hyperparams['resume'] = True
        
    print(f"Total epochs: {hyperparams['epochs']}")
    print(f"Hyperparameters: {hyperparams['optimizer']}(lr={hyperparams['lr0']}, momentum={hyperparams['momentum']})")
    print(f"Weight decay: {hyperparams['weight_decay']}, Label smoothing: 0.1")
    
    if use_advanced_techniques:
        print(f"Focal loss gamma: 2.5, Cosine LR schedule: {hyperparams['cos_lr']}")
        print("\nAdvanced Techniques Enabled:")
        print("- Coordinate Attention for spatial awareness")
        print("- Supervised Contrastive Loss for better class separation")
        print("- Enhanced Focal Loss (gamma=2.5)")
    else:
        print("\nRunning standard YOLOv8 training without advanced techniques")
    
    # Load the model
    model = YOLO(model_path)
    
    # Train the model
    if use_phased_training:
        # Adjust phases based on current epoch if resuming
        if current_epoch > 0:
            # Recalculate phase epochs based on current progress
            total_epochs = current_epoch + epochs  # Total epochs including those already completed
            phase1_end = int(total_epochs * 0.6)  # 60% of total epochs
            phase2_end = int(total_epochs * 0.8)  # 80% of total epochs
            
            # Determine which phase we're in
            if current_epoch < phase1_end:
                # Still in phase 1
                print(f"Resuming in Phase 1 (epochs {current_epoch}-{phase1_end})")
                phase1_params = get_phase_hyperparameters(hyperparams, phase=1, total_epochs=total_epochs)
                phase1_params['resume'] = True
                
                # Ensure we don't include invalid parameters
                for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
                    if param in phase1_params:
                        del phase1_params[param]
                
                # Disable early stopping for proper phases
                phase1_params['patience'] = 0
                
                # Continue phase 1 training
                results1 = model.train(**phase1_params)
                
                # Then continue with remaining phases
                last_checkpoint = str(Path(results1.save_dir) / 'weights' / 'last.pt')
                model = YOLO(last_checkpoint)
                
                # Phase 2
                print("\n" + "=" * 50)
                print(f"Phase 2: Medium Augmentation (epochs {phase1_end+1}-{phase2_end})")
                print("=" * 50)
                
                phase2_params = get_phase_hyperparameters(hyperparams, phase=2, total_epochs=total_epochs)
                
                # Ensure we don't include invalid parameters
                for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
                    if param in phase2_params:
                        del phase2_params[param]
                
                # Disable early stopping for proper phases
                phase2_params['patience'] = 0
                
                results2 = model.train(**phase2_params)
                
                # Phase 3
                print("\n" + "=" * 50)
                print(f"Phase 3: Freezing Backbone (epochs {phase2_end+1}-{total_epochs})")
                print("=" * 50)
                
                last_checkpoint2 = str(Path(results2.save_dir) / 'weights' / 'last.pt')
                model = YOLO(last_checkpoint2)
                phase3_params = get_phase_hyperparameters(hyperparams, phase=3, total_epochs=total_epochs)
                
                # Ensure we don't include invalid parameters
                for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
                    if param in phase3_params:
                        del phase3_params[param]
                
                # Disable early stopping for proper phases
                phase3_params['patience'] = 0
                
                results = model.train(**phase3_params)
                
            elif current_epoch < phase2_end:
                # In phase 2
                print(f"Resuming in Phase 2 (epochs {current_epoch}-{phase2_end})")
                phase2_params = get_phase_hyperparameters(hyperparams, phase=2, total_epochs=total_epochs)
                phase2_params['resume'] = True
                
                # Ensure we don't include invalid parameters
                for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
                    if param in phase2_params:
                        del phase2_params[param]
                
                # Disable early stopping for proper phases
                phase2_params['patience'] = 0
                
                results2 = model.train(**phase2_params)
                
                # Phase 3
                print("\n" + "=" * 50)
                print(f"Phase 3: Freezing Backbone (epochs {phase2_end+1}-{total_epochs})")
                print("=" * 50)
                
                last_checkpoint2 = str(Path(results2.save_dir) / 'weights' / 'last.pt')
                model = YOLO(last_checkpoint2)
                phase3_params = get_phase_hyperparameters(hyperparams, phase=3, total_epochs=total_epochs)
                
                # Ensure we don't include invalid parameters
                for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
                    if param in phase3_params:
                        del phase3_params[param]
                
                # Disable early stopping for proper phases
                phase3_params['patience'] = 0
                
                results = model.train(**phase3_params)
                
            else:
                # In phase 3
                print(f"Resuming in Phase 3 (epochs {current_epoch}-{total_epochs})")
                phase3_params = get_phase_hyperparameters(hyperparams, phase=3, total_epochs=total_epochs)
                phase3_params['resume'] = True
                
                # Ensure we don't include invalid parameters
                for param in ['callbacks', 'fl_gamma', 'cutmix', 'random_erasing']:
                    if param in phase3_params:
                        del phase3_params[param]
                
                # Disable early stopping for proper phases
                phase3_params['patience'] = 0
                
                results = model.train(**phase3_params)
        else:
            # Normal phased training starting from the beginning
            results = train_with_phases(
                model=model,
                hyperparams=hyperparams,
                epochs=hyperparams['epochs'],
                use_advanced_techniques=use_advanced_techniques
            )
    else:
        # Single phase training
        if use_advanced_techniques:
            # Create enhanced model with improvements
            model = create_enhanced_model(
                model_path=model_path,
                use_coordattn=True,
                use_contrastive_loss=True
            )
            
            # Remove callbacks from hyperparams as they're not directly supported
            if 'callbacks' in hyperparams:
                del hyperparams['callbacks']
    
        # Train with hyperparameters
        results = model.train(**hyperparams)
    
    # Generate advanced statistics if enabled
    if generate_advanced_statistics:
        print("\n" + "=" * 50)
        print("GENERATING ADVANCED MODEL STATISTICS")
        print("=" * 50)
        
        from modules.detection_metrics import generate_comprehensive_model_report
        
        # Run validation to get results
        validation_results = model.val()
        
        # Collect training history from results
        training_history = {
            'epochs': list(range(1, hyperparams['epochs'] + 1)),
            'train_metrics': {},
            'val_metrics': {}
        }
        
        # Extract metrics from results if available
        if hasattr(results, 'keys'):
            metrics_keys = [k for k in results.keys() if k not in ['maps', 'times', 'fitness']]
            for k in metrics_keys:
                if isinstance(results[k], list):
                    training_history['train_metrics'][k] = results[k]
        
        # Get model info
        model_info = {
            'model_name': name,
            'model_size_mb': 0,  # Initialize with zero
            'use_advanced_techniques': use_advanced_techniques,
            'use_phased_training': use_phased_training,
            'epochs': hyperparams['epochs'],
            'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else 0
        }
        
        # Try to get model size from last.pt if it exists
        try:
            last_model_path = str(Path(results.save_dir) / 'weights' / 'last.pt')
            if os.path.exists(last_model_path):
                model_info['model_size_mb'] = os.path.getsize(last_model_path) / (1024 * 1024)
        except Exception as e:
            print(f"Warning: Could not get model size - {e}")
        
        # Generate the report
        report_dir = os.path.join('runs', project, name, 'statistics')
        report = generate_comprehensive_model_report(
            validation_results=validation_results,
            training_history=training_history,
            model_info=model_info,
            output_dir=report_dir,
            use_wandb=use_wandb
        )
        
        print(f"\nComprehensive model statistics report generated in: {report_dir}")
        print(f"Summary report: {report['summary_path']}")
        print(f"HTML report: {report['html_report_path']}")
        
    # Get the final model
    if use_phased_training:
        # The final model is already returned by train_with_phases
        return results
    else:
        # For standard single-phase training, return best model
        last_checkpoint = str(Path(results.save_dir) / 'weights' / 'last.pt')
        model = YOLO(last_checkpoint)
        return model

def apply_semi_supervised_expansion(
    model: YOLO, 
    unlabeled_data_path: str, 
    output_path: str, 
    conf_threshold: float = 0.7,
    iou_threshold: float = 0.5,
) -> int:
    from tqdm import tqdm
    
    print(f"\nGenerating pseudo-labels from {unlabeled_data_path}")
    print(f"Using confidence threshold: {conf_threshold}")
    
    os.makedirs(output_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(unlabeled_data_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    count = 0
    class_counter = Counter()
    
    for img_file in tqdm(image_files, desc="Generating pseudo-labels"):
        img_path = os.path.join(unlabeled_data_path, img_file)
        
        results = model.predict(img_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            img = results[0].orig_img
            height, width = img.shape[:2]
            
            label_file = os.path.join(output_path, os.path.splitext(img_file)[0] + '.txt')
            
            with open(label_file, 'w') as f:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    # Update class counter
                    class_counter[cls_id] += 1
                    
                    # Convert xyxy to normalized xywh format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Normalize coordinates
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")
            
            count += 1
    
    # Log class distribution
    print(f"\nGenerated pseudo-labels for {count}/{len(image_files)} images")
    print("\nClass distribution in pseudo-labels:")
    print("-" * 50)
    
    # Get class names from the model if possible
    class_names = {}
    try:
        data_yaml_path = getattr(model, 'data', 'data.yaml')
        with open(data_yaml_path, 'r') as f:
            import yaml
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml.get('names', {})
    except:
        # If we can't get names, use numeric IDs
        pass
    
    # Display distribution table
    total_instances = sum(class_counter.values())
    print(f"{'Class':<20} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 50)
    
    for cls_id, cls_count in sorted(class_counter.items()):
        cls_name = class_names.get(cls_id, f"Class {cls_id}")
        percentage = (cls_count / total_instances) * 100 if total_instances > 0 else 0
        print(f"{cls_name:<20} | {cls_count:<8} | {percentage:.2f}%")
    
    # Convert class IDs to names for better visualization
    named_counter = {class_names.get(cls_id, f"Class {cls_id}"): count 
                     for cls_id, count in class_counter.items()}
    
    # Save distribution to a CSV file for later analysis
    distribution_path = os.path.join(output_path, "class_distribution.csv")
    with open(distribution_path, 'w') as f:
        f.write("class_id,class_name,count,percentage\n")
        for cls_id, cls_count in sorted(class_counter.items()):
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
            percentage = (cls_count / total_instances) * 100 if total_instances > 0 else 0
            f.write(f"{cls_id},{cls_name},{cls_count},{percentage:.2f}\n")
    
    print(f"\nClass distribution saved to {distribution_path}")
    
    return count

def generate_model_statistics(
    model_path: str,
    project: str = "vessel_detection",
    name: str = None,
    use_wandb: bool = False,
    data_yaml: str = "data.yaml"
) -> Dict[str, Any]:
    """
    Generate comprehensive statistics for a trained YOLO model.
    
    Args:
        model_path: Path to the model weights file (.pt)
        project: Project name for organizing results
        name: Name for this statistics run
        use_wandb: Whether to use Weights & Biases for logging
        data_yaml: Path to the data.yaml file
        
    Returns:
        Dictionary containing paths to various statistics outputs
    """
    # Load the model
    model = YOLO(model_path)
    
    # Generate a validation report
    results = model.val(data=data_yaml, project=project, name=name)
    
    # Extract the output directory from validation results
    output_dir = Path(results.save_dir)
    html_report_path = output_dir / "report.html"
    
    # Return paths to generated statistics
    return {
        "results": results,
        "output_dir": str(output_dir),
        "html_report_path": str(html_report_path)
    }

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Train YOLOv8s model for vessel detection with advanced techniques")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases tracking')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='Disable Weights & Biases tracking')
    parser.add_argument('--name', type=str, default="yolov8s_advanced", help='Run name')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--advanced', action='store_true', help='Enable advanced techniques')
    parser.add_argument('--no-advanced', dest='advanced', action='store_false', help='Disable advanced techniques')
    parser.add_argument('--phased', action='store_true', help='Use phased training approach')
    parser.add_argument('--no-phased', dest='phased', action='store_false', help='Use standard training')
    parser.add_argument('--unlabeled', type=str, default=None, help='Path to unlabeled data for semi-supervised expansion')
    parser.add_argument('--no-stats', dest='stats', action='store_false', help='Disable advanced statistics generation')
    
    parser.set_defaults(wandb=True, advanced=True, phased=True, stats=True)
    args = parser.parse_args()
    
    results = train_yolo(
        epochs=args.epochs,
        name=args.name,
        use_wandb=args.wandb,
        use_advanced_techniques=args.advanced,
        use_phased_training=args.phased,
        custom_hyperparams={'batch': args.batch, 'lr0': args.lr},
        generate_advanced_statistics=args.stats
    )
    
    if args.unlabeled and os.path.exists(args.unlabeled):
        model = YOLO(results.best)
        apply_semi_supervised_expansion(
            model=model,
            unlabeled_data_path=args.unlabeled,
            output_path="pseudo_labels"
        )
        print("\nPseudo-labels generated in 'pseudo_labels' directory")
        print("You can use these for additional training with semi-supervised learning") 