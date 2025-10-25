#!/usr/bin/env python3
"""Test script to verify GPU usage when DEVICE_MAP=auto"""

import torch
import psutil
import os
from configs import Config

def check_gpu_availability():
    """Check if CUDA is available and show GPU info"""
    print("=== GPU Availability Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print()

def show_current_config():
    """Display current device configuration"""
    print("=== Current Configuration ===")
    print(f"DEVICE_MAP: {Config.DEVICE_MAP}")
    print(f"TORCH_DTYPE: {Config.TORCH_DTYPE}")
    print()

def test_simple_tensor():
    """Test tensor placement with current device_map setting"""
    print("=== Simple Tensor Test ===")
    
    if torch.cuda.is_available():
        print("GPU memory before tensor creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # Create a tensor and see where it goes with auto device
    if Config.DEVICE_MAP == "auto" and torch.cuda.is_available():
        device = "cuda"
        print("DEVICE_MAP=auto and CUDA available, using GPU")
    elif Config.DEVICE_MAP == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print("DEVICE_MAP=cuda, using GPU")
    else:
        device = "cpu"
        print(f"Using CPU (DEVICE_MAP={Config.DEVICE_MAP})")
    
    # Create test tensor
    x = torch.randn(1000, 1000).to(device)
    print(f"Test tensor device: {x.device}")
    
    if torch.cuda.is_available() and x.device.type == "cuda":
        print(f"\nGPU memory after tensor creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print("✅ GPU is being used!")
    else:
        print("ℹ️  Using CPU")

if __name__ == "__main__":
    check_gpu_availability()
    show_current_config()
    test_simple_tensor()