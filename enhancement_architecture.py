import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

class CoordinateAttention(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class EnhancedDetectionModel(nn.Module):

    
    def __init__(self, model, use_coordattn=True, use_contrastive_loss=True):
        super(EnhancedDetectionModel, self).__init__()
        self.model = model
        self.use_coordattn = use_coordattn
        self.use_contrastive_loss = use_contrastive_loss
        self.class_distribution = Counter()
        self.log_frequency = 10  # Log distribution every 10 batches
        self.batch_count = 0
        self.has_class_names = False
        self.class_names = {}
        
        if self.use_contrastive_loss:
            from YOLOv8.modules.losses import SupConLoss
            self.sup_con_loss = SupConLoss(temperature=0.07)
            
        # Store original forward method
        self.original_forward = model.forward
        
        # Replace forward method with our enhanced version
        if self.use_coordattn or self.use_contrastive_loss:
            self.model.forward = self.enhanced_forward
    
    def add_coordinate_attention(self):
        from ultralytics.nn.modules import C2f
        
        model = self.model.model
        
        # Get the backbone layers where we'll add CA modules
        backbone_indices = []
        ca_modules = []
        
        # Find C2f blocks in the backbone to add CA modules after them
        for i, m in enumerate(model):
            if isinstance(m, C2f) and i > 0 and i < len(model) - 3:
                # Get the output channels of the C2f block
                # Try different methods to get output channels
                out_channels = None
                
                # Method 1: Check cv3 or cv2 attributes
                if hasattr(m, 'cv3') and hasattr(m.cv3, 'out_channels'):
                    out_channels = m.cv3.out_channels
                elif hasattr(m, 'cv2') and hasattr(m.cv2, 'out_channels'):
                    out_channels = m.cv2.out_channels
                
                # Method 2: Check if we can access input/output channels directly
                elif hasattr(m, 'c2'):
                    out_channels = m.c2
                elif hasattr(m, 'out_channels'):
                    out_channels = m.out_channels
                
                # Method 3: Fallback to examining previous or next layer
                elif i+1 < len(model) and hasattr(model[i+1], 'in_channels'):
                    out_channels = model[i+1].in_channels
                elif i > 0 and hasattr(model[i-1], 'out_channels'):
                    out_channels = model[i-1].out_channels
                
                # Method 4: Last resort - use a fixed value (this should be improved)
                else:
                    print(f"Warning: Could not determine channels for C2f module at index {i}")
                    # Skip this layer
                    continue
                
                # Create a CA module with matching channels
                ca = CoordinateAttention(out_channels, out_channels)
                ca_modules.append((i+1, ca))  # Insert after the C2f block
                backbone_indices.append(i)
        
        # Insert CA modules into the model
        offset = 0
        for idx, ca_module in ca_modules:
            model.insert(idx + offset, ca_module)
            offset += 1
            
        print(f"Added {len(ca_modules)} Coordinate Attention modules to the model")
        
        # Update the model
        self.model.model = model
        return model
    
    def enhanced_forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.original_forward(x, augment=True, profile=profile, visualize=visualize)
            
        # Forward pass through the model
        features, pred = self.original_forward(x, augment=False, profile=profile, visualize=visualize)
        
        # We'll store additional info in this dict
        additional_info = {}
        
        if self.training and self.use_contrastive_loss:
            # Extract backbone features for contrastive loss
            backbone_features = features[-1]  # Using the last layer features
            
            # Store for later use in loss computation
            additional_info['backbone_features'] = backbone_features
        
        return features, pred, additional_info
    
    def compute_contrastive_loss(self, features, targets):
        if not self.use_contrastive_loss:
            return 0.0
            
        # Extract class labels from targets
        batch_size = len(targets)
        labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        
        # Update class distribution counter
        batch_class_counter = Counter()
        
        for i, target in enumerate(targets):
            if len(target) > 0:  # If there are any targets
                # Use the most common class as the image label
                cls_ids = target[:, 1].long()  # Class IDs column
                # Get most common class (mode)
                unique_cls, counts = torch.unique(cls_ids, return_counts=True)
                most_common_idx = counts.argmax()
                label_id = unique_cls[most_common_idx].item()
                labels[i] = label_id
                
                # Update counter
                batch_class_counter[label_id] += 1
                
                # Also count all classes, not just the most common one per image
                for cls_id in cls_ids:
                    self.class_distribution[cls_id.item()] += 1
        
        # Compute contrastive loss
        features = features.view(batch_size, 1, -1)  # [B, 1, D]
        loss = self.sup_con_loss(features, labels)
        
        # Periodically log class distribution
        self.batch_count += 1
        
        return loss 
    
    def _log_class_distribution(self):
        if not self.class_distribution:
            return
            
        # Try to load class names if we haven't done so already
        if not self.has_class_names:
            try:
                # Load the data.yaml file to get class names
                import yaml
                with open('data.yaml', 'r') as f:
                    data_config = yaml.safe_load(f)
                    self.class_names = data_config.get('names', {})
                self.has_class_names = True
            except Exception as e:
                print(f"Could not load class names: {e}")
        
        # Create visualization of class distribution
        labels = []
        values = []
        
        for class_id, count in self.class_distribution.most_common():
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            labels.append(class_name)
            values.append(count)
        
        # Create simple bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save locally instead of to wandb
        plt.savefig('class_distribution.png')
        plt.close()
        
        # Create data for text output
        data = []
        for class_id, count in self.class_distribution.most_common():
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            data.append([class_name, count])
        
        # Print summary to console
        print("\nClass Distribution:")
        for row in data:
            print(f"{row[0]}: {row[1]}")
        print() 