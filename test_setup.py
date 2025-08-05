# Create test_setup.py
import cv2
import numpy as np
import tkinter as tk
from ultralytics import YOLO
import PIL
import matplotlib.pyplot as plt

print("✅ All packages imported successfully!")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")

# Test YOLO
try:
    model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ YOLO error: {e}")
