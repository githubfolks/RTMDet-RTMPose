"""
RTMDet Inference Script
Run detection on images using trained model weights.

Usage:
    python3 infer_det.py --image path/to/image.jpg
    python3 infer_det.py --image path/to/image.jpg --weights train/weights/rtmdet_custom.pth
"""

import torch
import cv2
import numpy as np
import argparse
import os
from models.rtmdet import RTMDet
from utils.box_ops import multiclass_nms

def preprocess(img, target_size=(640, 640)):
    """Resize and normalize image for inference."""
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, target_size)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
    return img_tensor, (h, w)

def decode_predictions(cls_scores, bbox_preds, strides, img_size, original_size, conf_thr=0.3, nms_thr=0.45):
    """Decode model outputs to bounding boxes."""
    from utils.box_ops import distance2bbox
    
    all_scores = []
    all_boxes = []
    
    for i, stride in enumerate(strides):
        cls_feat = cls_scores[i]  # (1, C, H, W)
        reg_feat = bbox_preds[i]  # (1, 4, H, W)
        
        B, C, H, W = cls_feat.shape
        
        # Generate anchor centers
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        cx = (x.reshape(-1) + 0.5) * stride
        cy = (y.reshape(-1) + 0.5) * stride
        anchors = torch.stack([cx, cy], dim=-1)  # (N, 2)
        
        # Reshape predictions
        cls_feat = cls_feat[0].permute(1, 2, 0).reshape(-1, C).sigmoid()  # (N, C)
        reg_feat = reg_feat[0].permute(1, 2, 0).reshape(-1, 4)  # (N, 4) - already exp'd in forward
        
        # Decode boxes
        boxes = distance2bbox(anchors, reg_feat * stride)
        
        all_scores.append(cls_feat)
        all_boxes.append(boxes)
    
    all_scores = torch.cat(all_scores, dim=0)  # (N_total, C)
    all_boxes = torch.cat(all_boxes, dim=0)    # (N_total, 4)
    
    # NMS
    boxes, scores, labels = multiclass_nms(all_boxes, all_scores, conf_thr, nms_thr)
    
    # Scale boxes to original image size
    scale_x = original_size[1] / img_size[0]
    scale_y = original_size[0] / img_size[1]
    
    if len(boxes) > 0:
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y
    
    return boxes, scores, labels

def draw_boxes(img, boxes, scores, labels, class_names=None):
    """Draw bounding boxes on image."""
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        
        # Color based on confidence
        color = (0, int(255 * score), int(255 * (1 - score)))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{int(label)}: {score:.2f}"
        if class_names and int(label) < len(class_names):
            label_text = f"{class_names[int(label)]}: {score:.2f}"
        
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='RTMDet Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='train/weights/rtmdet_custom.pth', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: input_det.jpg)')
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image {args.image}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    # Load model
    print(f"Loading model from {args.weights}...")
    model = RTMDet('s', num_classes=80)  # 80 classes as per training default (even if data has 1)
    
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        print("Weights loaded successfully.")
    else:
        print(f"Warning: Weights file not found at {args.weights}. Using random initialization.")
    
    model.eval()
    
    # Preprocess
    img_tensor, original_size = preprocess(img_rgb)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        cls_scores, bbox_preds = model(img_tensor)
    
    # Decode
    boxes, scores, labels = decode_predictions(
        cls_scores, bbox_preds, 
        strides=[8, 16, 32],
        img_size=(640, 640),
        original_size=original_size,
        conf_thr=args.conf,
        nms_thr=args.nms
    )
    
    print(f"Detected {len(boxes)} objects.")
    
    # Draw
    result = draw_boxes(img.copy(), boxes.numpy(), scores.numpy(), labels.numpy(), class_names=['person'])
    
    # Save
    output_path = args.output or args.image.replace('.', '_det.')
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
