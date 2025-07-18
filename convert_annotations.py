#!/usr/bin/env python3
"""
Convert CSV annotations to YOLO format for training
"""

import pandas as pd
import os
from pathlib import Path

def convert_csv_to_yolo(csv_path, output_dir, image_width=512, image_height=512):
    """
    Convert CSV annotations to YOLO format
    
    Args:
        csv_path: Path to the CSV annotation file
        output_dir: Directory to save YOLO annotations
        image_width: Width of images (default: 512)
        image_height: Height of images (default: 512)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Group by filename to handle multiple annotations per image
    grouped = df.groupby('filename')
    
    for filename, group in grouped:
        # Create YOLO annotation file path
        base_name = os.path.splitext(filename)[0]
        yolo_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(yolo_file, 'w') as f:
            for _, row in group.iterrows():
                # Convert bounding box coordinates to YOLO format
                x_center = (row['xmin'] + row['xmax']) / 2 / image_width
                y_center = (row['ymin'] + row['ymax']) / 2 / image_height
                width = (row['xmax'] - row['xmin']) / image_width
                height = (row['ymax'] - row['ymin']) / image_height
                
                # Class ID (0 for ball)
                class_id = 0
                
                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Converted {filename} -> {yolo_file}")

def main():
    """Convert all annotation files"""
    
    # Convert train annotations
    print("Converting train annotations...")
    convert_csv_to_yolo('train/_annotations.csv', 'train')
    
    # Convert validation annotations
    print("Converting validation annotations...")
    convert_csv_to_yolo('valid/_annotations.csv', 'valid')
    
    # Convert test annotations
    print("Converting test annotations...")
    convert_csv_to_yolo('test/_annotations.csv', 'test')
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 