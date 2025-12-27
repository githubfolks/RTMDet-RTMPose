"""
Visualize keypoints with confidence thresholding.
- Green: High confidence (>0.6)
- Yellow: Medium confidence (0.4-0.6)
- Red: Low confidence (<0.4) - optionally hidden
"""
import torch
import cv2
import numpy as np
import os
import glob
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from infer import preprocess_det, postprocess_det

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO Skeleton
SKELETON_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]

def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    """Decode SimCC output with confidence scores."""
    N, K, W_bins = simcc_x.shape
    
    locs_x = torch.argmax(simcc_x, dim=2).float()
    locs_y = torch.argmax(simcc_y, dim=2).float()
    
    # Confidence from softmax
    conf_x = torch.max(torch.softmax(simcc_x, dim=2), dim=2)[0]
    conf_y = torch.max(torch.softmax(simcc_y, dim=2), dim=2)[0]
    confidence = (conf_x + conf_y) / 2
    
    locs_x /= simcc_split_ratio
    locs_y /= simcc_split_ratio
    
    return torch.stack([locs_x, locs_y], dim=-1), confidence

def get_color_by_confidence(conf, high_thresh=0.6, med_thresh=0.4):
    """Return color based on confidence level."""
    if conf >= high_thresh:
        return (0, 255, 0)  # Green - high confidence
    elif conf >= med_thresh:
        return (0, 255, 255)  # Yellow - medium confidence
    else:
        return (0, 0, 255)  # Red - low confidence

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_weights = 'train/weights/rtmdet_custom.pth'
    pose_weights = 'train/weights/rtmpose_custom.pth'
    input_dir = 'sample_input'
    output_dir = 'comparison'
    
    # Confidence thresholds
    CONF_THRESHOLD = 0.3  # Hide keypoints below this
    HIGH_CONF = 0.6
    MED_CONF = 0.4
    
    # Load Models
    print(f"Loading RTMDet from {det_weights}...")
    det_model = RTMDet('s').to(device)
    det_model.load_state_dict(torch.load(det_weights, map_location=device))
    det_model.eval()
    
    print(f"Loading RTMPose from {pose_weights}...")
    pose_model = RTMPose('s', input_size=(256, 192)).to(device)
    pose_model.load_state_dict(torch.load(pose_weights, map_location=device))
    pose_model.eval()
    
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nConfidence thresholds:")
    print(f"  Hide below: {CONF_THRESHOLD}")
    print(f"  Green (high): >= {HIGH_CONF}")
    print(f"  Yellow (medium): >= {MED_CONF}")
    print(f"  Red (low): < {MED_CONF}")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape
        
        # Detection
        det_input, orig_shape = preprocess_det(img)
        det_input = det_input.to(device)
        
        with torch.no_grad():
            cls_scores, bbox_preds = det_model(det_input)
            
        bboxes, scores, labels = postprocess_det(cls_scores, bbox_preds, (640, 640), orig_shape, 0.16, nms_thr=0.1)
        
        vis_img = img.copy()
        padding_ratio = 0.25
        
        # Collect valid boxes
        valid_boxes = []
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
            
            if score < 0.3:
                if x1 < -2 or y1 < -2 or x2 > w_img + 2 or y2 > h_img + 2:
                    continue
            
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
            
            valid_boxes.append((x1_refined, y1_refined, x2_refined, y2_refined, float(score)))
            cv2.rectangle(vis_img, (x1_refined, y1_refined), (x2_refined, y2_refined), (255, 0, 0), 2)
        
        # Process each person
        for person_idx, (x1_box, y1_box, x2_box, y2_box, det_score) in enumerate(valid_boxes):
            w_box = x2_box - x1_box
            h_box = y2_box - y1_box
            
            pad_w = int(w_box * padding_ratio)
            pad_h = int(h_box * padding_ratio)
            
            x1_crop = max(0, x1_box - pad_w)
            y1_crop = max(0, y1_box - pad_h)
            x2_crop = min(w_img, x2_box + pad_w)
            y2_crop = min(h_img, y2_box + pad_h)
            
            crop_w = x2_crop - x1_crop
            crop_h = y2_crop - y1_crop
            
            if crop_w < 10 or crop_h < 10:
                continue
            
            person_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
            target_size = (192, 256)
            pose_input_img = cv2.resize(person_crop, target_size)
            
            pose_input_img_rgb = cv2.cvtColor(pose_input_img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(pose_input_img_rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                simcc_x, simcc_y = pose_model(tensor)
            
            keypoints, confidence = decode_simcc(simcc_x, simcc_y)
            keypoints = keypoints[0].cpu().numpy()
            confidence = confidence[0].cpu().numpy()
            
            scale_x = crop_w / 192.0
            scale_y = crop_h / 256.0
            
            # Transform keypoints and store with confidence
            trans_keypoints = []
            for i, (pt, conf) in enumerate(zip(keypoints, confidence)):
                kx = int(pt[0] * scale_x) + x1_crop
                ky = int(pt[1] * scale_y) + y1_crop
                trans_keypoints.append((kx, ky, conf))
            
            # Draw keypoints (only above threshold)
            for i, (kx, ky, conf) in enumerate(trans_keypoints):
                if conf >= CONF_THRESHOLD:
                    color = get_color_by_confidence(conf, HIGH_CONF, MED_CONF)
                    cv2.circle(vis_img, (kx, ky), 5, color, -1)
                    cv2.circle(vis_img, (kx, ky), 5, (0, 0, 0), 1)  # Black border
            
            # Draw skeleton (only if both keypoints are above threshold)
            for (u, v) in SKELETON_LINKS:
                if u < len(trans_keypoints) and v < len(trans_keypoints):
                    pt1_x, pt1_y, conf1 = trans_keypoints[u]
                    pt2_x, pt2_y, conf2 = trans_keypoints[v]
                    
                    if conf1 >= CONF_THRESHOLD and conf2 >= CONF_THRESHOLD:
                        # Use the lower confidence for line color
                        min_conf = min(conf1, conf2)
                        color = get_color_by_confidence(min_conf, HIGH_CONF, MED_CONF)
                        cv2.line(vis_img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
            
            # Add person label
            cv2.putText(vis_img, f"P{person_idx+1}: {det_score:.2f}", 
                       (x1_box, y1_box - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_img, "Confidence Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(vis_img, (20, legend_y + 20), 6, (0, 255, 0), -1)
        cv2.putText(vis_img, f"High (>={HIGH_CONF})", (35, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(vis_img, (20, legend_y + 40), 6, (0, 255, 255), -1)
        cv2.putText(vis_img, f"Med ({MED_CONF}-{HIGH_CONF})", (35, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(vis_img, (20, legend_y + 60), 6, (0, 0, 255), -1)
        cv2.putText(vis_img, f"Low (<{MED_CONF})", (35, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        save_path = os.path.join(output_dir, f"conf_{filename}")
        cv2.imwrite(save_path, vis_img)
        print(f"Processed {filename} -> {save_path}")

if __name__ == "__main__":
    main()
