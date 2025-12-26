import torch
import cv2
import numpy as np
import os
import glob
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from infer import preprocess_det, postprocess_det

def get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):
    """
    Simpler affine transform implementation.
    Output size: (w, h)
    """
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def box_to_center_scale(box, model_input_size=(256, 192)):
    # box: x1, y1, x2, y2
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
    scale = np.array([w, h], dtype=np.float32)
    
    # Keep aspect ratio
    aspect_ratio = model_input_size[1] / model_input_size[0] # h/w = 0.75
    # RTMPose default input (192, 256) (W, H) or (256, 192)?
    # RTMPose config: input_size=(256, 192) usually means H=256, W=192
    
    # Checking models/rtmpose.py: input_size=(256, 192). SimCC bins w=input[1]*2 => 192*2, h=input[0]*2 => 256*2
    # So H=256, W=192.
    
    # Adjust scale to keep aspect ratio
    # target aspect: 192/256 = 0.75
    if w / h > (192.0 / 256.0):
        scale[1] = w * 256.0 / 192.0
    else:
        scale[0] = h * 192.0 / 256.0
        
    scale = scale * 1.25 # Padding
    return center, scale

def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    """
    Decode SimCC output to keypoint coordinates.
    simcc_x: (N, K, W_bins)
    simcc_y: (N, K, H_bins)
    """
    # Simple argmax decoding
    N, K, W_bins = simcc_x.shape
    _, H_bins = simcc_y.shape[0], simcc_y.shape[2]
    
    locs_x = torch.argmax(simcc_x, dim=2).float() # (N, K)
    locs_y = torch.argmax(simcc_y, dim=2).float() # (N, K)
    
    locs_x /= simcc_split_ratio
    locs_y /= simcc_split_ratio
    
    # We might need gaussian refinement but argmax is good baseline
    return torch.stack([locs_x, locs_y], dim=-1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_weights = 'train/weights/rtmdet_custom.pth'
    pose_weights = 'train/weights/rtmpose_custom.pth' # Assuming exists
    input_dir = 'sample_input'
    output_dir = 'sample_output'
    score_thr = 0.16 # Optimized
    nms_thr = 0.1
    
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
    
    # COCO Skeleton
    skeleton_links = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Face
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12), # Torso
        (11, 13), (12, 14), (13, 15), (14, 16) # Legs
    ]
    keypoint_colors = [(0,255,0)] * 17 
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape
        
        # 1. Detection
        det_input, orig_shape = preprocess_det(img)
        det_input = det_input.to(device)
        
        with torch.no_grad():
            cls_scores, bbox_preds = det_model(det_input)
            
        bboxes, scores, labels = postprocess_det(cls_scores, bbox_preds, (640, 640), orig_shape, score_thr, nms_thr=nms_thr)
        
        # Visualize Detections
        vis_img = img.copy()
        
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
            
            # Filter edge
            if score < 0.3:
                 if x1 < -2 or y1 < -2 or x2 > w_img + 2 or y2 > h_img + 2:
                      continue
            
            # Refine Box (1.5x width, 20% right scale)
            w = x2 - x1
            center_x = x1 + w / 2
            
            # Shift
            shift_x = w * 0.2
            center_x += shift_x
            
            # Expand
            new_w = w * 1.5
            x1 = int(center_x - new_w / 2)
            x2 = int(center_x + new_w / 2)
            
            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # 2. Pose Estimation - WHOLE IMAGE STRATEGY
        # We run pose once per image, not per box, because training was on whole images.
        
        # Resize to 192x256
        target_size = (192, 256) # W, H
        pose_input_img = cv2.resize(img, target_size)
        
        # Preprocess: RGB, / 255.0
        pose_input_img_rgb = cv2.cvtColor(pose_input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(pose_input_img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            simcc_x, simcc_y = pose_model(tensor)
        
        keypoints = decode_simcc(simcc_x, simcc_y)
        keypoints = keypoints[0].cpu().numpy() # (17, 2) in 192x256
        
        # Map back to original image
        scale_x = w_img / 192.0
        scale_y = h_img / 256.0
        
        # Visualize Keypoints (Red dots, Green lines)
        trans_keypoints = []
        for pt in keypoints:
            kx = int(pt[0] * scale_x)
            ky = int(pt[1] * scale_y)
            trans_keypoints.append((kx, ky))
            cv2.circle(vis_img, (kx, ky), 4, (0, 0, 255), -1)
            
        for (u, v) in skeleton_links:
            if u < len(trans_keypoints) and v < len(trans_keypoints):
                pt1 = trans_keypoints[u]
                pt2 = trans_keypoints[v]
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, vis_img)
        print(f"Processed {filename}")

if __name__ == "__main__":
    main()
