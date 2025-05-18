import os
import torch
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO
from pathlib import Path

# Import from modules_initalization
from YOLOv8.modules.modules_initalization import (
    CoordinateAttention, 
    EnhancedDetectionModel,
    SupConLoss
)

def create_enhanced_model(model_path, use_coordattn=True, use_contrastive_loss=True):
    # Load the base YOLO model
    model = YOLO(model_path)
    
    # Create enhanced model wrapper
    enhanced_model = EnhancedDetectionModel(model.model, 
                                          use_coordattn=use_coordattn,
                                          use_contrastive_loss=use_contrastive_loss)
    
    # Add coordinate attention modules
    if use_coordattn:
        enhanced_model.add_coordinate_attention()
    
    # Copy trained weights from original model to our enhanced model
    model.model = enhanced_model
    
    return model

def create_safe_transition_callback():
    """
    This function creates callbacks for safely transitioning between training phases.
    Note: These callbacks are not directly used with the model.train() function anymore.
    Instead, they should be used with a custom training loop if needed.
    """
    def on_train_end(trainer):
        # Save the last model weights explicitly to ensure we have a checkpoint
        save_dir = Path(trainer.save_dir) / 'weights'
        save_dir.mkdir(exist_ok=True, parents=True)
        last_model_path = save_dir / 'last.pt'
        
        # Make sure we always have a last.pt file
        if trainer.best is not None:
            try:
                if hasattr(trainer.metrics, 'best'):
                    # Use the best model if metrics.best exists
                    trainer.model.save(file=last_model_path)
                    print(f"Saved best model to {last_model_path}")
                else:
                    # Fallback to saving current model state
                    trainer.model.save(file=last_model_path)
                    print(f"Saved current model state to {last_model_path}")
            except Exception as e:
                # If any error occurs, save the current model state
                print(f"Error accessing metrics.best: {e}")
                trainer.model.save(file=last_model_path)
                print(f"Saved current model state to {last_model_path}")
        else:
            # Save current model state if best is None
            trainer.model.save(file=last_model_path)
            print(f"Saved current model state to {last_model_path}")
    
    # Create callback dictionary
    callbacks = {
        'on_train_end': on_train_end
    }
    
    return callbacks

def create_training_callbacks(use_contrastive_loss=True, contrastive_weight=0.1):
    """
    This function creates callbacks for training with advanced techniques.
    Note: These callbacks are not directly used with the model.train() function anymore.
    Instead, they should be used with a custom training loop if needed.
    """
    contrastive_loss_values = []
    
    def on_train_start(trainer):
        print("Starting training with enhanced model...")
        if use_contrastive_loss:
            print("Using Supervised Contrastive Loss with weight:", contrastive_weight)
    
    def on_train_batch_end(trainer):
        if use_contrastive_loss and hasattr(trainer.model, 'compute_contrastive_loss'):
            # Get the last batch's features and targets
            features = trainer.batch_memory.get('backbone_features', None)
            targets = trainer.batch_memory.get('targets', None)
            
            if features is not None and targets is not None:
                # Compute contrastive loss
                con_loss = trainer.model.compute_contrastive_loss(features, targets)
                
                # Scale and add to the main loss
                trainer.loss += contrastive_weight * con_loss
                
                # Store for logging
                contrastive_loss_values.append(con_loss.item())
    
    def on_train_epoch_end(trainer):
        if use_contrastive_loss and contrastive_loss_values:
            # Log average contrastive loss for the epoch
            avg_con_loss = sum(contrastive_loss_values) / len(contrastive_loss_values)
            print(f"Epoch {trainer.epoch}: Contrastive Loss: {avg_con_loss:.4f}")
            contrastive_loss_values.clear()
    
    # Create callback dictionary
    callbacks = {
        'on_train_start': on_train_start,
        'on_train_batch_end': on_train_batch_end,
        'on_train_epoch_end': on_train_epoch_end
    }
    
    return callbacks 