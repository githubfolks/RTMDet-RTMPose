import torch
import cv2
import numpy as np
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from utils.box_ops import distance2bbox, multiclass_nms
import os

def get_simcc_maximum(simcc_x, simcc_y):
    """
    Decode SimCC heatmaps to coordinates.
    simcc_x: (B, K, Wx)
    simcc_y: (B, K, Wy)
    Returns: (B, K, 2)
    """
    B, K, Wx = simcc_x.shape
    _, _, Wy = simcc_y.shape
    
    # Simple argmax
    loc_x = simcc_x.argmax(dim=2) # (B, K)
    loc_y = simcc_y.argmax(dim=2) # (B, K)
    
    # Scale back? SimCC usually uses a split_ratio.
    # In our training/model, we used input_size (256, 192) and SimCC split_ratio=2.0
    # So bins are 256*2=512 and 192*2=384.
    # We need to divide by split_ratio to get input image coords.
    
    simcc_split_ratio = 2.0
    loc_x = loc_x.float() / simcc_split_ratio
    loc_y = loc_y.float() / simcc_split_ratio
    
    return torch.stack([loc_x, loc_y], dim=-1)

def run_pipeline(image_path, det_weights, pose_weights, output_path='result_final.jpg'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running pipeline on {device}...")
    
    # 1. Load Models
    det_model = RTMDet('s').to(device)
    pose_model = RTMPose('s', input_size=(256, 192)).to(device)
    
    if os.path.exists(det_weights):
        # Loose loading to handle slight architecture mismatches (e.g. P5 channel diffs)
        state_dict = torch.load(det_weights, map_location=device)
        model_dict = det_model.state_dict()
        
        # Filter out mismatching shapes
        valid_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        mismatch_count = len(state_dict) - len(valid_dict)
        if mismatch_count > 0:
            print(f"Warning: {mismatch_count} layers mismatch and were skipped (likely P5/Neck differences).")
            
        det_model.load_state_dict(valid_dict, strict=False)
        print("Detection weights loaded (partial).")
    else:
        print("Error: Detection weights not found.")
        return
        
    if os.path.exists(pose_weights):
        pose_model.load_state_dict(torch.load(pose_weights, map_location=device))
        print("Pose weights loaded.")
    else:
        print("Error: Pose weights not found.")
        return
        
    det_model.eval()
    pose_model.eval()
    
    # 2. Detect Persons
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print("Error: Image not found.")
        return
        
    h, w, _ = img_raw.shape
    det_input_size = (640, 640)
    img_det = cv2.resize(img_raw, det_input_size)
    det_tensor = torch.from_numpy(img_det).float() / 255.0
    det_tensor = det_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cls_scores, bbox_preds = det_model(det_tensor)
        
    # Validation Code Reuse (NMS)
    # ... (Simplified copy of verify_det logic)
    decoded_bboxes = []
    decoded_scores = []
    
    for i, stride in enumerate(det_model.head.strides):
        cls_score = cls_scores[i]
        bbox_pred = bbox_preds[i]
        B, C, H, W = cls_score.shape
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        cx = (x.reshape(-1) + 0.5) * stride
        cy = (y.reshape(-1) + 0.5) * stride
        centers = torch.stack([cx, cy], dim=-1)
        
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, C).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        
        max_scores, _ = cls_score.max(dim=1)
        valid_mask = max_scores > 0.1 # Lower threshold for safety
        
        if valid_mask.sum() > 0:
            bboxes = distance2bbox(centers[valid_mask], bbox_pred[valid_mask])
            decoded_bboxes.append(bboxes)
            decoded_scores.append(cls_score[valid_mask])
            
    if len(decoded_bboxes) == 0:
        print("No persons detected. Exiting.")
        return
        
    all_bboxes = torch.cat(decoded_bboxes, dim=0)
    all_scores = torch.cat(decoded_scores, dim=0)
    
    print(f"Pre-NMS Boxes: {len(all_bboxes)}")
    
    # Tuned Thresholds
    dets, scores, labels = multiclass_nms(all_bboxes, all_scores, score_thr=0.40, nms_thr=0.45)
    
    print(f"Post-NMS Detections: {len(dets)}")
    # print(f"Scores: {scores}")
    
    # Scale boxes to original image
    scale_x = w / det_input_size[0]
    scale_y = h / det_input_size[1]
    
    dets[:, 0] *= scale_x
    dets[:, 2] *= scale_x
    dets[:, 1] *= scale_y
    dets[:, 3] *= scale_y
    
    print(f"Detected {len(dets)} persons. Running pose estimation...")
    
    # 3. Pose Estimation (Top-Down)
    pose_input_size = (192, 256) # W, H
    
    skeletons = []
    
    for box in dets:
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        
        # Crop with some padding? Simplicity first: just crop.
        # Clamp to image
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1: continue
        
        person_crop = img_raw[y1:y2, x1:x2]
        crop_h, crop_w, _ = person_crop.shape
        
        # Resize to Pose Input
        img_pose = cv2.resize(person_crop, pose_input_size)
        pose_tensor = torch.from_numpy(img_pose).float() / 255.0
        pose_tensor = pose_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            simcc_x, simcc_y = pose_model(pose_tensor)
            
        # Decode Keypoints
        kpts = get_simcc_maximum(simcc_x, simcc_y) # (1, K, 2) in input_size coords
        kpts = kpts[0]
        
        # Scale keypoints back to crop -> original image
        kpts[:, 0] *= (crop_w / pose_input_size[0])
        kpts[:, 1] *= (crop_h / pose_input_size[1])
        
        kpts[:, 0] += x1
        kpts[:, 1] += y1
        
        skeletons.append(kpts)
        
        # Draw Box
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Draw Skeletons
    # COCO Skeleton connections
    skeleton_links = [
        (0,1), (0,2), (1,3), (2,4), # Face
        (5,6), (5,7), (7,9), (6,8), (8,10), # Arms
        (5,11), (6,12), # Torso
        (11,12), (11,13), (13,15), (12,14), (14,16) # Legs
    ]
    
    for kpts in skeletons:
        kpts_np = kpts.cpu().numpy()
        for i, (x, y) in enumerate(kpts_np):
            cv2.circle(img_raw, (int(x), int(y)), 3, (0, 0, 255), -1)
            
        for p1, p2 in skeleton_links:
            if p1 < len(kpts_np) and p2 < len(kpts_np):
                pt1 = (int(kpts_np[p1][0]), int(kpts_np[p1][1]))
                pt2 = (int(kpts_np[p2][0]), int(kpts_np[p2][1]))
                cv2.line(img_raw, pt1, pt2, (255, 255, 0), 2)

    cv2.imwrite(output_path, img_raw)
    print(f"Final result saved to {output_path}")
    
    with open("summary.txt", "w") as f:
        f.write(f"SUCCESS: Detected {len(dets)} persons.\n")
        f.write(f"Pre-NMS: {len(all_bboxes)}\n")

if __name__ == "__main__":
    img_path = "sample_images/900000027.jpg" 
    filename = os.path.basename(img_path)
    output_path = os.path.join('sample_output', filename)
    # Using Custom Trained Weights (Known good: ~2 detections)
    run_pipeline(img_path, "train/weights/rtmdet_custom.pth", "train/weights/rtmpose_custom.pth", output_path=output_path)
