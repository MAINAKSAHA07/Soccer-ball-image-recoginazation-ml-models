#!/usr/bin/env python3
"""
Train a lightweight YOLOv8 model for ball detection
"""

import os
import yaml
from ultralytics import YOLO
import torch

def train_ball_detection_model():
    """
    Train a lightweight YOLOv8 model for ball detection
    """
    
    print("🚀 Starting ball detection model training...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a lightweight YOLOv8 model (nano size for speed)
    model = YOLO('yolov8n.pt')  # nano model for lightweight inference
    
    # Training configuration - using only valid YOLOv8 parameters
    training_config = {
        'data': 'dataset_config.yaml',
        'epochs': 50,  # Moderate training for good performance
        'imgsz': 512,  # Image size
        'batch': 16,   # Batch size
        'device': device,
        'workers': 4,  # Number of workers for data loading
        'patience': 10,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'cache': False,  # Don't cache images to save memory
        'optimizer': 'AdamW',  # Use AdamW optimizer
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,  # Momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,  # Masks should overlap during training
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Use dropout regularization
        'val': True,  # Validate during training
        'plots': True,  # Generate training plots
    }
    
    print("📊 Training configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n🎯 Starting training...")
    results = model.train(**training_config)
    
    print("✅ Training completed!")
    
    # Save the trained model
    model_path = 'ball_detection_model.pt'
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    return model, results

def evaluate_model(model):
    """
    Evaluate the trained model on test set
    """
    print("\n📈 Evaluating model on test set...")
    
    # Run validation on test set
    results = model.val(data='dataset_config.yaml', split='test')
    
    print("✅ Evaluation completed!")
    return results

if __name__ == "__main__":
    # Train the model
    model, training_results = train_ball_detection_model()
    
    # Evaluate the model
    evaluation_results = evaluate_model(model)
    
    print("\n🎉 Ball detection model training and evaluation completed!")
    print("📁 Model files:")
    print("  - ball_detection_model.pt (PyTorch model)")
    print("  - runs/detect/train/ (Training logs and plots)")
    print("  - runs/detect/val/ (Validation results)") 