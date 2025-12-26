import torch
import sys
import os

print(f"Initial: {torch.cuda.is_available()}", flush=True)

try:
    from models.rtmdet import RTMDet
    print(f"After models.rtmdet: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"Error importing RTMDet: {e}")

try:
    from data.yolo import YOLODataset
    print(f"After data.yolo: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"Error importing YOLODataset: {e}")
    
try:
    import cv2
    print(f"After cv2: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"Error importing cv2: {e}")

print(f"Final: {torch.cuda.is_available()}", flush=True)
