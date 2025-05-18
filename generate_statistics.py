import os
import sys
import argparse
from pathlib import Path

# Add current directory to path if needed
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from YOLOv8.train import generate_model_statistics

def main():
    # Hardcoded model path - modify this to point to your model file
    MODEL_PATH = "/Users/atharvabadkas/Coding /YOLO Vessel Detection /YOLO V2 Model/vessel_detection/yolov8s_improved/weights/best.pt"
    
    # Hardcoded output location in YOLOv8 folder
    OUTPUT_DIR = "/Users/atharvabadkas/Coding /YOLO Vessel Detection /YOLO V2 Model/YOLOv8/model_stats"
    
    parser = argparse.ArgumentParser(description="Generate comprehensive statistics for a YOLO model")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the model weights file (e.g., best.pt or last.pt)')
    parser.add_argument('--project', type=str, default=OUTPUT_DIR, help='Project directory for saving results')
    parser.add_argument('--name', type=str, default=None, help='Name for this statistics run (default: auto-generated from model filename)')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases integration')
    parser.add_argument('--data', type=str, default="data.yaml", help='Path to data.yaml file')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.project, exist_ok=True)
    
    # Auto-generate run name if not provided
    if args.name is None:
        model_filename = os.path.basename(args.model_path)
        args.name = f"stats_{os.path.splitext(model_filename)[0]}"
    
    # Generate statistics
    print(f"Generating statistics for model: {args.model_path}")
    print(f"Results will be saved to: {args.project}/{args.name}")
    
    try:
        report = generate_model_statistics(
            model_path=args.model_path,
            project=args.project,
            name=args.name,
            use_wandb=args.wandb,
            data_yaml=args.data
        )
        
        # Print the summary
        print("\nStatistics generation completed successfully!")
        print(f"HTML report: {report['html_report_path']}")
        return 0
    except Exception as e:
        print(f"Error generating statistics: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 