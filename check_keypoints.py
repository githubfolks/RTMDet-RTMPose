"""
Script to check and compare keypoints from model output.
Shows raw model keypoints vs transformed keypoints.
"""
import torch
import cv2
import numpy as np
import os
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from infer import preprocess_det, postprocess_det

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    """
    Decode SimCC output to keypoint coordinates.
    Also return confidence scores based on softmax values.
    """
    N, K, W_bins = simcc_x.shape
    _, _, H_bins = simcc_y.shape
    
    # Get argmax positions
    locs_x = torch.argmax(simcc_x, dim=2).float()
    locs_y = torch.argmax(simcc_y, dim=2).float()
    
    # Get confidence as max softmax value
    conf_x = torch.max(torch.softmax(simcc_x, dim=2), dim=2)[0]
    conf_y = torch.max(torch.softmax(simcc_y, dim=2), dim=2)[0]
    confidence = (conf_x + conf_y) / 2
    
    locs_x /= simcc_split_ratio
    locs_y /= simcc_split_ratio
    
    return torch.stack([locs_x, locs_y], dim=-1), confidence

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_weights = 'train/weights/rtmdet_custom.pth'
    pose_weights = 'train/weights/rtmpose_custom.pth'
    
    # Test on one image
    test_image = 'sample_input/200000346.jpg'
    
    # Load Models
    print(f"Loading RTMDet from {det_weights}...")
    det_model = RTMDet('s').to(device)
    det_model.load_state_dict(torch.load(det_weights, map_location=device))
    det_model.eval()
    
    print(f"Loading RTMPose from {pose_weights}...")
    pose_model = RTMPose('s', input_size=(256, 192)).to(device)
    pose_model.load_state_dict(torch.load(pose_weights, map_location=device))
    pose_model.eval()
    
    # Load image
    img = cv2.imread(test_image)
    h_img, w_img, _ = img.shape
    print(f"\n{'='*60}")
    print(f"Image: {test_image}")
    print(f"Original image size: {w_img} x {h_img}")
    print(f"{'='*60}")
    
    # Detection
    det_input, orig_shape = preprocess_det(img)
    det_input = det_input.to(device)
    
    with torch.no_grad():
        cls_scores, bbox_preds = det_model(det_input)
        
    bboxes, scores, labels = postprocess_det(cls_scores, bbox_preds, (640, 640), orig_shape, 0.16, nms_thr=0.1)
    
    print(f"\nDetected {len(bboxes)} persons")
    
    # Process each person
    padding_ratio = 0.25
    person_id = 0
    
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
        
        # Filter low confidence edge detections
        if score < 0.3:
            if x1 < -2 or y1 < -2 or x2 > w_img + 2 or y2 > h_img + 2:
                continue
        
        # Refine Box
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w / 2
        shift_x = w * 0.2
        center_x += shift_x
        new_w = w * 1.5
        x1_refined = int(center_x - new_w / 2)
        x2_refined = int(center_x + new_w / 2)
        padding_y = int(h * 0.1)
        y1_refined = y1 - padding_y
        y2_refined = y2 + padding_y
        
        person_id += 1
        print(f"\n{'='*60}")
        print(f"Person {person_id} (Detection score: {float(score):.3f})")
        print(f"{'='*60}")
        print(f"Detection bbox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Refined bbox: [{x1_refined}, {y1_refined}, {x2_refined}, {y2_refined}]")
        
        # Calculate crop region
        w_box = x2_refined - x1_refined
        h_box = y2_refined - y1_refined
        pad_w = int(w_box * padding_ratio)
        pad_h = int(h_box * padding_ratio)
        
        x1_crop = max(0, x1_refined - pad_w)
        y1_crop = max(0, y1_refined - pad_h)
        x2_crop = min(w_img, x2_refined + pad_w)
        y2_crop = min(h_img, y2_refined + pad_h)
        
        crop_w = x2_crop - x1_crop
        crop_h = y2_crop - y1_crop
        
        print(f"Crop region: [{x1_crop}, {y1_crop}, {x2_crop}, {y2_crop}]")
        print(f"Crop size: {crop_w} x {crop_h}")
        
        if crop_w < 10 or crop_h < 10:
            print("Crop too small, skipping...")
            continue
        
        # Crop and resize
        person_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        target_size = (192, 256)  # W, H
        pose_input_img = cv2.resize(person_crop, target_size)
        
        print(f"Model input size: {target_size[0]} x {target_size[1]} (W x H)")
        
        # Preprocess
        pose_input_img_rgb = cv2.cvtColor(pose_input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(pose_input_img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        
        # Run pose model
        with torch.no_grad():
            simcc_x, simcc_y = pose_model(tensor)
        
        print(f"\nSimCC output shapes:")
        print(f"  simcc_x: {simcc_x.shape} (N, K, W_bins)")
        print(f"  simcc_y: {simcc_y.shape} (N, K, H_bins)")
        
        keypoints, confidence = decode_simcc(simcc_x, simcc_y)
        keypoints = keypoints[0].cpu().numpy()  # (17, 2) in model input space
        confidence = confidence[0].cpu().numpy()  # (17,)
        
        print(f"\n--- Raw Model Output (in 192x256 input space) ---")
        print(f"{'Keypoint':<15} {'X':>8} {'Y':>8} {'Conf':>8}")
        print("-" * 42)
        for i, (name, (x, y), conf) in enumerate(zip(COCO_KEYPOINT_NAMES, keypoints, confidence)):
            print(f"{name:<15} {x:>8.2f} {y:>8.2f} {conf:>8.3f}")
        
        # Transform keypoints back to original image
        scale_x = crop_w / 192.0
        scale_y = crop_h / 256.0
        
        print(f"\n--- Transform parameters ---")
        print(f"Scale X: {scale_x:.4f}")
        print(f"Scale Y: {scale_y:.4f}")
        print(f"Offset: ({x1_crop}, {y1_crop})")
        
        print(f"\n--- Transformed Keypoints (in original image space) ---")
        print(f"{'Keypoint':<15} {'X':>8} {'Y':>8} {'Conf':>8}")
        print("-" * 42)
        for i, (name, (x, y), conf) in enumerate(zip(COCO_KEYPOINT_NAMES, keypoints, confidence)):
            tx = int(x * scale_x) + x1_crop
            ty = int(y * scale_y) + y1_crop
            print(f"{name:<15} {tx:>8} {ty:>8} {conf:>8.3f}")

if __name__ == "__main__":
    main()
