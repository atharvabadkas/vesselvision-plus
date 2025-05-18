import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union

def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    # Generate colors based on class ID to ensure consistency
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Green (dark)
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128)   # Teal
    ]
    
    # Use modulo to handle more classes than colors
    return colors[class_id % len(colors)]

def draw_detection_boxes(
    image: np.ndarray,
    boxes: List,
    class_names: List[str],
    high_quality: bool = True
) -> np.ndarray:
    annotated_image = image.copy()
    
    if high_quality:
        # Higher quality rendering
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id]
            
            # Thicker line for better visibility
            thickness = 3
            
            # Draw box with custom color based on class
            color = get_color_for_class(cls_id)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Add class label and confidence
            label = f"{cls_name} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background at the bottom of the box
            cv2.rectangle(
                annotated_image, 
                (x1, y2), 
                (x1 + text_size[0], y2 + text_size[1] + 5), 
                color, 
                -1
            )
            
            # Add text at the bottom
            cv2.putText(
                annotated_image, 
                label, 
                (x1, y2 + text_size[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
    
    return annotated_image

def plot_detection_stats(stats: Dict[str, Any], output_path: str = None) -> None:
    class_stats = stats.get('class_statistics', {})
    
    if not class_stats:
        print("No class statistics to plot")
        return
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Class counts
    class_names = list(class_stats.keys())
    class_counts = [stats["count"] for stats in class_stats.values()]
    
    # Sort by count
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_counts = [class_counts[i] for i in sorted_indices]
    
    axs[0].bar(sorted_names, sorted_counts, color='skyblue')
    axs[0].set_title('Detection Count by Class')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Confidence levels
    avg_confidences = [stats["avg_confidence"] for stats in class_stats.values()]
    min_confidences = [stats["min_confidence"] for stats in class_stats.values()]
    max_confidences = [stats["max_confidence"] for stats in class_stats.values()]
    
    # Sort by the same order as counts
    sorted_avg = [avg_confidences[i] for i in sorted_indices]
    sorted_min = [min_confidences[i] for i in sorted_indices]
    sorted_max = [max_confidences[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_names))
    width = 0.25
    
    axs[1].bar(x - width, sorted_avg, width, label='Avg Confidence', color='green')
    axs[1].bar(x, sorted_min, width, label='Min Confidence', color='red')
    axs[1].bar(x + width, sorted_max, width, label='Max Confidence', color='blue')
    
    axs[1].set_title('Confidence Levels by Class')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Confidence')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(sorted_names, rotation=45)
    axs[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        print(f"Statistics plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_class_weights(class_weights: Dict[str, float], output_path: str = None) -> None:
    if not class_weights:
        print("No class weights to plot")
        return
    
    # Sort by weight (descending)
    sorted_classes = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
    classes = [c[0] for c in sorted_classes]
    weights = [c[1] for c in sorted_classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, weights, color='purple')
    
    # Add value annotations on top of bars
    for bar, weight in zip(bars, weights):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{weight:.2f}",
            ha='center',
            fontsize=9
        )
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Neutral Weight')
    
    plt.title('Class Weights for Training')
    plt.xlabel('Class')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        print(f"Class weights plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close() 