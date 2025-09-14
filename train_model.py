#!/usr/bin/env python3
"""
Script to train the MNIST CNN model
Run this script from the project root directory
"""

import sys
import os
from pathlib import Path

# Add the models directory to Python path
sys.path.append(str(Path(__file__).parent / "code" / "models"))

from train_model import train_model

if __name__ == "__main__":
    print("Starting MNIST CNN training...")
    print("This will download the MNIST dataset if not already present.")
    print("Training will take a few minutes...")

    try:
        model, accuracy = train_model()
        print(f"\n✅ Training completed successfully!")
        print(f"Final test accuracy: {accuracy:.2f}%")
        print("Model saved to models/ directory")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
