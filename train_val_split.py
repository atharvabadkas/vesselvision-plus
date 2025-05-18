import os
import shutil
import random

def create_yolo_folder_structure():
    # Define paths
    source_dir = "/Users/atharvabadkas/Coding /YOLO Vessel Detection /YOLO V2 Model/labeled_dataset_25"
    dest_dir = "data"
    paths = {
        'source_images': os.path.join(source_dir, "images"),
        'source_labels': os.path.join(source_dir, "labels"),
        'train_images': os.path.join(dest_dir, "train", "images"),
        'train_labels': os.path.join(dest_dir, "train", "labels"),
        'val_images': os.path.join(dest_dir, "val", "images"),
        'val_labels': os.path.join(dest_dir, "val", "labels")
    }
    
    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Get and split image files
    image_files = [f for f in os.listdir(paths['source_images']) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise Exception(f"No image files found in {paths['source_images']}")
    
    # Split into train (80%) and validation (20%) sets    
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files, val_files = image_files[:split_idx], image_files[split_idx:]
    
    def copy_files(files, src_img, src_label, dst_img, dst_label):
        for img_file in files:
            # Copy image if it exists
            src_img_path = os.path.join(src_img, img_file)
            if not os.path.exists(src_img_path):
                print(f"Warning: Image file {img_file} not found")
                continue
                
            shutil.copy(src_img_path, os.path.join(dst_img, img_file))
            
            # Copy corresponding label if it exists
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(src_label, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dst_label, label_file))
            else:
                print(f"Warning: Label file {label_file} not found for {img_file}")
    
    # Process train and validation sets
    copy_files(train_files, paths['source_images'], paths['source_labels'], 
              paths['train_images'], paths['train_labels'])
    copy_files(val_files, paths['source_images'], paths['source_labels'], 
              paths['val_images'], paths['val_labels'])

    # Print summary
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Validation images: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
    print(f"Data directory structure created at: {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    create_yolo_folder_structure() 