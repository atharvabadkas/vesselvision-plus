import os
import cv2
import yaml
import torch
import sys
import time
import numpy as np
from pathlib import Path
import heapq
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
from ultralytics import YOLO

# Add the parent directory to sys.path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom utilities using modules_initalization
from modules.modules_initalization import (
    EnhancedDetectionModel,
    get_color_for_class,
    draw_detection_boxes,
    calculate_detection_statistics,
    generate_class_weights,
    apply_test_time_augmentation
)

class VesselDetector:
    
    def __init__(self, model_path: str, data_yaml_path: str = 'data.yaml', use_enhanced_model: bool = True):

        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.use_enhanced_model = use_enhanced_model
        
        try:
            if use_enhanced_model:
                from model_enhancement import create_enhanced_model
                
                self.model = create_enhanced_model(
                    model_path=model_path,
                    use_coordattn=True,
                    use_contrastive_loss=False
                )
                print(f"Enhanced model loaded with Coordinate Attention from {model_path}")
            else:
                self.model = YOLO(model_path)
                
            self.model.to(self.device)
            print(f"Using device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
            
        try:
            with open(data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                self.class_names = data_yaml['names']
            print(f"Loaded {len(self.class_names)} classes from {data_yaml_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading class names: {e}")
    
    def get_top_n_predictions(self, boxes, n: int = 3) -> List[Tuple[str, float, int]]:
        predictions = []
        for box in boxes:
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = self.class_names[class_id]
            predictions.append((class_name, conf, class_id))
        
        return heapq.nlargest(n, predictions, key=lambda x: x[1])
    
    def process_image(self, 
                     image: np.ndarray, 
                     target_width: int = 640, 
                     use_tta: bool = False,
                     high_quality: bool = False) -> Tuple[np.ndarray, List[Tuple[str, float, int]]]:
        if image is None:
            raise ValueError("Image is None")
        
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)
        resized_image = cv2.resize(image, (target_width, target_height))
        
        if use_tta:
            results = apply_test_time_augmentation(
                model=self.model,
                image=resized_image,
                device=self.device
            )
            print("Using Test-Time Augmentation for higher accuracy")
        else:
            results = self.model.predict(
                resized_image, 
                conf=0.001,  # Set very low confidence to get all predictions
                device=self.device, 
                verbose=False
            )
        
        annotated_image = resized_image.copy()
        top_predictions = []
        
        if len(results) > 0:
            result = results[0]
            
            if len(result.boxes) > 0:
                # Get top 3 predictions for terminal display
                top_predictions = self.get_top_n_predictions(result.boxes, n=3)
                
                # Keep only the highest confidence box for display
                highest_conf_idx = torch.argmax(result.boxes.conf).item()
                boxes_to_keep = torch.tensor([highest_conf_idx])
                result.boxes = result.boxes[boxes_to_keep]
            
            if high_quality:
                annotated_image = draw_detection_boxes(
                    annotated_image, 
                    result.boxes, 
                    self.class_names,
                    high_quality=True
                )
            else:
                annotated_image = result.plot()
                
        return annotated_image, top_predictions
    
    def process_image_file(self, 
                          image_path: str, 
                          output_dir: str = None,
                          target_width: int = 640,
                          use_tta: bool = False,
                          high_quality: bool = False) -> Dict[str, Any]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            annotated_image, predictions = self.process_image(
                image, 
                target_width=target_width,
                use_tta=use_tta,
                high_quality=high_quality
            )
            
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, annotated_image)
            
            # Print top 3 predictions to terminal
            print(f"\nTop predictions for {os.path.basename(image_path)}:")
            for i, (class_name, confidence, class_id) in enumerate(predictions[:3], 1):
                print(f"  {i}. {class_name}: {confidence:.4f}")
                
            return {
                "filename": os.path.basename(image_path),
                "predictions": predictions,
                "num_predictions": len(predictions)
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                "filename": os.path.basename(image_path),
                "error": str(e),
                "predictions": [],
                "num_predictions": 0
            }
    
    def process_image_folder(self, 
                            image_folder: str, 
                            output_dir: str = "output",
                            target_width: int = 640,
                            use_tta: bool = False,
                            high_quality: bool = True,
                            save_json: bool = False) -> Dict[str, Any]:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(image_folder).glob(f"*{ext}")))
            image_files.extend(list(Path(image_folder).glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return {"error": "No images found", "processed": 0}
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        results = []
        for image_path in tqdm(image_files, desc=f"Processing {len(image_files)} images"):
            result = self.process_image_file(
                image_path=str(image_path),
                output_dir=output_dir,
                target_width=target_width,
                use_tta=use_tta,
                high_quality=high_quality
            )
            results.append(result)
        
        stats = calculate_detection_statistics(results)
        
        if save_json:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "detection_results.json")
            
            json_output = {
                "summary": {
                    "total_images": stats["total_images"],
                    "successful_processes": stats["successful_processes"],
                    "total_predictions": stats["total_predictions"],
                    "predictions_per_image": stats["predictions_per_image"],
                    "accuracy_percentage": stats["accuracy_percentage"]
                },
                "class_statistics": stats["class_statistics"],
                "image_results": []
            }
            
            for r in results:
                json_output["image_results"].append({
                    "filename": r.get("filename", ""),
                    "predictions": [
                        {"class": p[0], "confidence": p[1], "class_id": p[2]}
                        for p in r.get("predictions", [])
                    ],
                    "error": r.get("error", None)
                })
                
            with open(json_path, 'w') as f:
                json.dump(json_output, f, indent=2)
                
            print(f"Results saved to {json_path}")
            
            self.generate_class_weights(stats["class_statistics"], output_dir)
        
        return stats
    
    def generate_class_weights(self, class_stats: Dict, output_dir: str) -> None:
        normalized_weights = generate_class_weights(class_stats)
        
        if normalized_weights:
            weights_path = os.path.join(output_dir, "class_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(normalized_weights, f, indent=2)
                
            print(f"Class weights saved to {weights_path}")
            print("Use these weights in your next training run for better performance on weaker classes.")
            
            print("\nClass weights for next training:")
            print("-" * 50)
            print(f"{'Class':<20} | {'Weight':<10} | {'Count':<8} | {'Avg Conf':<8}")
            print("-" * 50)
            
            for cls in sorted(normalized_weights.keys()):
                print(f"{cls:<20} | {normalized_weights[cls]:<10.2f} | {class_stats[cls]['count']:<8} | {class_stats[cls]['avg_confidence']:<8.2f}")

def format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def main():
    print("\n" + "=" * 80)
    print("VESSEL DETECTION TESTING")
    print("=" * 80)
    
    # Hardcoded paths that can be easily modified
    DEFAULT_MODEL_PATH = "/Users/atharvabadkas/Coding /YOLO Vessel Detection /YOLO V2 Model/vessel_detection/yolov8s_improved/weights/best.pt"  # Change this to your model path
    DEFAULT_IMAGE_FOLDER = "/Users/atharvabadkas/Coding /YOLO Vessel Detection /YOLO V2 Model/20250320" # Change this to your image folder
    
    # Get user input for model and images
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        image_folder = sys.argv[2]
    else:
        print(f"Using default paths: {DEFAULT_MODEL_PATH} and {DEFAULT_IMAGE_FOLDER}")
        model_path = DEFAULT_MODEL_PATH
        image_folder = DEFAULT_IMAGE_FOLDER
    
    start_time = time.time()
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return False
    
    data_yaml = os.path.abspath('../data.yaml')
    if not os.path.exists(data_yaml):
        data_yaml = os.path.abspath('./data.yaml')
        if not os.path.exists(data_yaml):
            print(f"Error: data.yaml not found. Please place it in the project root directory.")
            return False
    
    print("\nTesting Configuration:")
    print(f"- Model: {model_path}")
    print(f"- Image folder: {image_folder}")
    print(f"- Using dataset config: {data_yaml}")
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"\nProcessing images in {image_folder}...")
        detector = VesselDetector(
            model_path=model_path,
            data_yaml_path=data_yaml,
            use_enhanced_model=True
        )
        
        stats = detector.process_image_folder(
            image_folder=image_folder,
            output_dir=output_dir,
            use_tta=False,
            high_quality=True,
            save_json=True
        )
        duration = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"TESTING COMPLETED in {format_time(duration)}")
        print(f"Detection accuracy: {stats['accuracy_percentage']:.2f}%")
        print(f"Results saved in: {os.path.abspath(output_dir)}")
        print("=" * 80)
        
        return stats
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        return False

if __name__ == "__main__":
    main() 