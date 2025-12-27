"""
Batch Pose Estimation with Multi-Person Detection and Pose Classification
Detects: sitting, standing, running, walking
"""
import torch
import cv2
import numpy as np
import os
import glob
import argparse
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from infer import preprocess_det, postprocess_det

# ============================================
# CONFIGURATION
# ============================================

# Confidence thresholds
CONF_HIDE_THRESHOLD = 0.5   # Hide keypoints below this
CONF_HIGH_THRESHOLD = 0.7   # Green - high confidence
CONF_MED_THRESHOLD = 0.6    # Yellow - medium confidence

# COCO Keypoint indices
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoint indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# COCO Skeleton
SKELETON_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]

# Pose colors for visualization
POSE_COLORS = {
    'sitting': (255, 165, 0),    # Orange
    'standing': (0, 255, 0),     # Green
    'running': (0, 0, 255),      # Red
    'walking': (255, 255, 0),    # Cyan
    'unknown': (128, 128, 128),  # Gray
}


# ============================================
# POSE CLASSIFICATION
# ============================================

def calculate_angle(p1, p2, p3):
    """
    Calculate angle at p2 formed by p1-p2-p3.
    Returns angle in degrees.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def classify_pose(keypoints, confidence, conf_threshold=0.015):
    """
    Classify pose based on keypoint positions and angles.
    
    Args:
        keypoints: List of (x, y, conf) tuples for 17 COCO keypoints
        confidence: Array of confidence scores
        conf_threshold: Minimum confidence to consider a keypoint
        conf_threshold: Minimum confidence to consider a keypoint
        
    Returns:
        pose: 'sitting', 'standing', 'running', 'walking', or 'unknown'
        details: Dictionary with analysis details
    """
    details = {}
    
    # Extract keypoint positions
    kpts = np.array([[kp[0], kp[1]] for kp in keypoints])
    confs = np.array([kp[2] for kp in keypoints])
    
    # Check if essential keypoints are visible
    essential_lower = [L_HIP, R_HIP, L_KNEE, R_KNEE]
    essential_upper = [L_SHOULDER, R_SHOULDER]
    
    lower_visible = sum(confs[i] >= conf_threshold for i in essential_lower)
    upper_visible = sum(confs[i] >= conf_threshold for i in essential_upper)
    
    if lower_visible < 2:
        return 'unknown', {'reason': 'insufficient lower body keypoints'}
    
    # ---- Calculate key measurements ----
    
    # Hip and knee positions (average of left and right if both visible)
    if confs[L_HIP] >= conf_threshold and confs[R_HIP] >= conf_threshold:
        hip_pos = (kpts[L_HIP] + kpts[R_HIP]) / 2
    elif confs[L_HIP] >= conf_threshold:
        hip_pos = kpts[L_HIP]
    else:
        hip_pos = kpts[R_HIP]
    
    if confs[L_KNEE] >= conf_threshold and confs[R_KNEE] >= conf_threshold:
        knee_pos = (kpts[L_KNEE] + kpts[R_KNEE]) / 2
    elif confs[L_KNEE] >= conf_threshold:
        knee_pos = kpts[L_KNEE]
    else:
        knee_pos = kpts[R_KNEE]
    
    # Calculate hip-knee vertical distance
    hip_knee_vert = knee_pos[1] - hip_pos[1]  # Positive = knee below hip
    
    # Calculate knee angle (hip-knee-ankle) for each leg
    knee_angles = []
    for hip_idx, knee_idx, ankle_idx in [(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)]:
        if all(confs[i] >= conf_threshold for i in [hip_idx, knee_idx, ankle_idx]):
            angle = calculate_angle(kpts[hip_idx], kpts[knee_idx], kpts[ankle_idx])
            knee_angles.append(angle)
    
    avg_knee_angle = np.mean(knee_angles) if knee_angles else None
    details['avg_knee_angle'] = avg_knee_angle if avg_knee_angle is not None else 180
    
    # Calculate hip-knee vertical distance (safe)
    hip_visible = confs[L_HIP] >= conf_threshold or confs[R_HIP] >= conf_threshold
    knee_visible = confs[L_KNEE] >= conf_threshold or confs[R_KNEE] >= conf_threshold
    
    hip_knee_vert = None
    if hip_visible and knee_visible:
        # Use visible points only or average if both
        h_ys = []
        if confs[L_HIP] >= conf_threshold: h_ys.append(kpts[L_HIP][1])
        if confs[R_HIP] >= conf_threshold: h_ys.append(kpts[R_HIP][1])
        hip_y = np.mean(h_ys)
        
        k_ys = []
        if confs[L_KNEE] >= conf_threshold: k_ys.append(kpts[L_KNEE][1])
        if confs[R_KNEE] >= conf_threshold: k_ys.append(kpts[R_KNEE][1])
        knee_y = np.mean(k_ys)
        
        hip_knee_vert = knee_y - hip_y
        details['hip_knee_vert'] = hip_knee_vert
    else:
        details['hip_knee_vert'] = 'N/A'
    
    # Calculate leg spread (horizontal distance between ankles)
    leg_spread = 0
    if confs[L_ANKLE] >= conf_threshold and confs[R_ANKLE] >= conf_threshold:
        leg_spread = abs(kpts[L_ANKLE][0] - kpts[R_ANKLE][0])
        details['leg_spread'] = leg_spread
    
    # Calculate body height approximation
    body_height = 1
    if upper_visible >= 1 and (confs[L_ANKLE] >= conf_threshold or confs[R_ANKLE] >= conf_threshold):
        shoulder_y = kpts[L_SHOULDER][1] if confs[L_SHOULDER] >= conf_threshold else kpts[R_SHOULDER][1]
        ankle_y = kpts[L_ANKLE][1] if confs[L_ANKLE] >= conf_threshold else kpts[R_ANKLE][1]
        body_height = max(abs(ankle_y - shoulder_y), 1)
        details['body_height'] = body_height
    elif hip_knee_vert is not None:
        # Fallback estimation if ankles missing but knees present
        body_height = max(abs(hip_knee_vert) * 2.5, 1)

    # Leg spread ratio
    spread_ratio = leg_spread / body_height if body_height > 0 else 0
    details['spread_ratio'] = spread_ratio
    
    # ---- Classification Logic ----
    
    # SITTING: Knees are bent significantly (angle < 120°) OR knees close to hips vertically
    # If angle is known, use it. If not, rely on vertical distance if reliable.
    is_sitting_angle = (avg_knee_angle is not None and avg_knee_angle < 130)
    is_sitting_vert = (hip_knee_vert is not None and hip_knee_vert < body_height * 0.1)
    
    if is_sitting_angle or (avg_knee_angle is None and is_sitting_vert):
         return 'sitting', details
    
    # Use default angle 180 if unknown for further checks
    check_angle = avg_knee_angle if avg_knee_angle is not None else 180
    
    # RUNNING: Wide leg spread + potentially bent arms
    if spread_ratio > 0.35:
        # Check if arms are bent (elbows)
        arm_bent = False
        for shoulder, elbow, wrist in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
            if all(confs[i] >= conf_threshold for i in [shoulder, elbow, wrist]):
                arm_angle = calculate_angle(kpts[shoulder], kpts[elbow], kpts[wrist])
                if arm_angle < 150:
                    arm_bent = True
                    break
        
        if arm_bent or spread_ratio > 0.45:
            return 'running', details
    
    # WALKING: Moderate leg spread
    if spread_ratio > 0.12:
        return 'walking', details
    
    # STANDING: Legs close together, upright posture
    if check_angle > 150 and spread_ratio <= 0.12:
        return 'standing', details
    
    # Default to standing if we can't determine otherwise
    if hip_knee_vert > body_height * 0.2:
        return 'standing', details
    
    return 'unknown', details


# ============================================
# HELPER FUNCTIONS
# ============================================

def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    """Decode SimCC output to keypoint coordinates with confidence scores."""
    N, K, W_bins = simcc_x.shape
    
    locs_x = torch.argmax(simcc_x, dim=2).float()
    locs_y = torch.argmax(simcc_y, dim=2).float()
    
    conf_x = torch.max(torch.softmax(simcc_x, dim=2), dim=2)[0]
    conf_y = torch.max(torch.softmax(simcc_y, dim=2), dim=2)[0]
    confidence = (conf_x + conf_y) / 2
    
    locs_x /= simcc_split_ratio
    locs_y /= simcc_split_ratio
    
    return torch.stack([locs_x, locs_y], dim=-1), confidence


def get_color_by_confidence(conf):
    """Return color based on confidence level (BGR format)."""
    if conf >= CONF_HIGH_THRESHOLD:
        return (0, 255, 0)  # Green
    elif conf >= CONF_MED_THRESHOLD:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red


# ============================================
# MAIN PROCESSING
# ============================================

def process_image(img, det_model, pose_model, device, 
                  score_thr=0.16, nms_thr=0.1, 
                  show_pose_label=True, show_keypoints=True):
    """
    Process a single image for multi-person pose estimation.
    
    Returns:
        vis_img: Visualization image
        results: List of dictionaries with detection and pose info for each person
    """
    h_img, w_img = img.shape[:2]
    
    # 1. Detection
    det_input, orig_shape = preprocess_det(img)
    det_input = det_input.to(device)
    
    with torch.no_grad():
        cls_scores, bbox_preds = det_model(det_input)
    
    bboxes, scores, labels = postprocess_det(
        cls_scores, bbox_preds, (640, 640), orig_shape, score_thr, nms_thr=nms_thr
    )
    
    vis_img = img.copy()
    results = []
    
    # Collect valid detection boxes
    valid_boxes = []
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
        
        # Filter edge detections with low confidence
        if score < 0.3:
            if x1 < -2 or y1 < -2 or x2 > w_img + 2 or y2 > h_img + 2:
                continue
        
        # Refine bounding box
        w, h = x2 - x1, y2 - y1
        center_x = x1 + w / 2 + w * 0.2  # Shift right
        new_w = w * 1.5
        
        x1_refined = int(center_x - new_w / 2)
        x2_refined = int(center_x + new_w / 2)
        y1_refined = y1 - int(h * 0.1)
        y2_refined = y2 + int(h * 0.1)
        
        valid_boxes.append({
            'bbox': (x1_refined, y1_refined, x2_refined, y2_refined),
            'score': float(score)
        })
    
    # 2. Pose Estimation for each person
    padding_ratio = 0.25
    
    for person_idx, box_info in enumerate(valid_boxes):
        x1_box, y1_box, x2_box, y2_box = box_info['bbox']
        det_score = box_info['score']
        
        w_box = x2_box - x1_box
        h_box = y2_box - y1_box
        
        # Add padding
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
        
        # Crop and resize
        person_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        pose_input = cv2.resize(person_crop, (192, 256))
        pose_input_rgb = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(pose_input_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            simcc_x, simcc_y = pose_model(tensor)
        
        keypoints, confidence = decode_simcc(simcc_x, simcc_y)
        keypoints = keypoints[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()
        
        # Transform keypoints to original image coordinates
        scale_x = crop_w / 192.0
        scale_y = crop_h / 256.0
        
        trans_keypoints = []
        for pt, conf in zip(keypoints, confidence):
            kx = int(pt[0] * scale_x) + x1_crop
            ky = int(pt[1] * scale_y) + y1_crop
            trans_keypoints.append((kx, ky, conf))
        
        # Classify pose
        pose_class, pose_details = classify_pose(trans_keypoints, confidence)
        
        # Store result
        results.append({
            'person_id': person_idx + 1,
            'bbox': box_info['bbox'],
            'det_score': det_score,
            'keypoints': trans_keypoints,
            'pose': pose_class,
            'pose_details': pose_details
        })
        
        # ---- Visualization ----
        
        # Draw bounding box with pose-specific color
        box_color = POSE_COLORS.get(pose_class, (128, 128, 128))
        cv2.rectangle(vis_img, (x1_box, y1_box), (x2_box, y2_box), box_color, 2)
        
        # Draw keypoints
        if show_keypoints:
            for kx, ky, conf in trans_keypoints:
                if conf >= CONF_HIDE_THRESHOLD:
                    color = get_color_by_confidence(conf)
                    cv2.circle(vis_img, (kx, ky), 5, color, -1)
                    cv2.circle(vis_img, (kx, ky), 5, (0, 0, 0), 1)
            
            # Draw skeleton
            for (u, v) in SKELETON_LINKS:
                pt1_x, pt1_y, conf1 = trans_keypoints[u]
                pt2_x, pt2_y, conf2 = trans_keypoints[v]
                
                if conf1 >= CONF_HIDE_THRESHOLD and conf2 >= CONF_HIDE_THRESHOLD:
                    min_conf = min(conf1, conf2)
                    color = get_color_by_confidence(min_conf)
                    cv2.line(vis_img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
        
        # Draw pose label
        if show_pose_label:
            label = f"P{person_idx+1}: {pose_class.upper()}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x = x1_box
            label_y = max(y1_box - 10, 20)
            
            # Background for label
            cv2.rectangle(vis_img, 
                         (label_x - 2, label_y - label_size[1] - 4),
                         (label_x + label_size[0] + 2, label_y + 4),
                         box_color, -1)
            cv2.putText(vis_img, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis_img, f"Detected: {len(results)} person(s)", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    legend_y += 25
    for pose_name, color in POSE_COLORS.items():
        if pose_name != 'unknown':
            cv2.rectangle(vis_img, (10, legend_y - 12), (25, legend_y + 3), color, -1)
            cv2.putText(vis_img, pose_name.capitalize(), (30, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            legend_y += 18
    
    return vis_img, results


def main():
    parser = argparse.ArgumentParser(description='Multi-Person Pose Estimation with Action Classification')
    parser.add_argument('--input', type=str, default='sample_input', help='Input directory or image path')
    parser.add_argument('--output', type=str, default='sample_output', help='Output directory')
    parser.add_argument('--det_weights', type=str, default='train/weights/rtmdet_custom.pth')
    parser.add_argument('--pose_weights', type=str, default='train/weights/rtmpose_best.pth')
    parser.add_argument('--score_thr', type=float, default=0.16)
    parser.add_argument('--nms_thr', type=float, default=0.1)
    parser.add_argument('--no_keypoints', action='store_true', help='Hide keypoint visualization')
    parser.add_argument('--no_pose_label', action='store_true', help='Hide pose classification label')
    args = parser.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading RTMDet from {args.det_weights}...")
    det_model = RTMDet('s').to(device)
    det_model.load_state_dict(torch.load(args.det_weights, map_location=device))
    det_model.eval()
    
    print(f"Loading RTMPose from {args.pose_weights}...")
    pose_model = RTMPose('s', input_size=(256, 192)).to(device)
    pose_model.load_state_dict(torch.load(args.pose_weights, map_location=device))
    pose_model.eval()
    
    # Get image paths
    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, '*.jpg'))
        image_paths += glob.glob(os.path.join(args.input, '*.png'))
    else:
        image_paths = [args.input]
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("-" * 50)
    
    # Statistics
    pose_counts = {'sitting': 0, 'standing': 0, 'running': 0, 'walking': 0, 'unknown': 0}
    total_persons = 0
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load: {filename}")
            continue
        
        vis_img, results = process_image(
            img, det_model, pose_model, device,
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            show_pose_label=not args.no_pose_label,
            show_keypoints=not args.no_keypoints
        )
        
        # Update statistics
        total_persons += len(results)
        for r in results:
            pose_counts[r['pose']] += 1
        
        # Print results
        print(f"\n{filename}:")
        for r in results:
            print(f"  Person {r['person_id']}: {r['pose'].upper()} (det_score: {r['det_score']:.2f})")
        
        # Save output
        save_path = os.path.join(args.output, filename)
        cv2.imwrite(save_path, vis_img)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total persons detected: {total_persons}")
    print(f"Pose breakdown:")
    for pose, count in pose_counts.items():
        if count > 0:
            pct = 100 * count / max(total_persons, 1)
            print(f"  {pose.capitalize():10s}: {count:3d} ({pct:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
