import torch
import torch.nn.functional as F

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Shape (n, 4), [l, t, r, b].
        max_shape (tuple): Shape (h, w).
    Returns:
        Tensor: Decoded bboxes (n, 4), [x1, y1, x2, y2].
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

def bbox_iou(box1, box2, eps=1e-7):
    """Calculate IoU of two sets of bboxes."""
    # Simplified for correct shapes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area + b2_area - inter_area + eps
    return inter_area / union

import torchvision

def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_thr, max_num=100):
    """NMS for multi-class bboxes.
    Args:
         multi_bboxes (Tensor): (N, 4) or (N, 4)
         multi_scores (Tensor): (N, C)
    """
    # Simply taking max score class for now (usually RTMDet is class-agnostic NMS or per-class)
    # Simplified implementation
    scores, labels = multi_scores.max(dim=1)
    valid_mask = scores > score_thr
    bboxes = multi_bboxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    if bboxes.numel() == 0:
        return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0)
    
    keep = torchvision.ops.nms(bboxes, scores, nms_thr) # Requires torchvision
    if max_num > 0:
        keep = keep[:max_num]
        
    return bboxes[keep], scores[keep], labels[keep]
