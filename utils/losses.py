import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import bbox_iou

def quality_focal_loss(pred, target, beta=2.0):
    """
    Quality Focal Loss (QFL) is a generalization of Focal Loss.
    pred: (N, C), raw logits (sigmoid NOT applied).
    target: (N, C), 0-1 quality (IoU) for positives, 0 for negatives.
    
    Uses stable BCE computation.
    """
    assert pred.shape == target.shape
    
    # Apply sigmoid
    pred_sigmoid = pred.sigmoid()
    
    # QFL: -|y - p|^beta * BCE
    # Scale factor: |y - p|^beta where y is target (0 or IoU), p is predicted prob
    scale_factor = (pred_sigmoid.detach() - target).abs().pow(beta)
    
    # Use stable BCE with logits
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    loss = scale_factor * bce
    
    return loss.sum()

def giou_loss(pred, target, eps=1e-7):
    """
    GIoU Loss.
    pred: (N, 4) in xyxy
    target: (N, 4) in xyxy
    """
    iou = bbox_iou(pred, target, eps=eps)
    
    # Enclosing box
    pred_x1, pred_y1, pred_x2, pred_y2 = pred.chunk(4, -1)
    target_x1, target_y1, target_x2, target_y2 = target.chunk(4, -1)
    
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)
    
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
    
    # GIoU = IoU - (Enc - Union) / Enc
    # Loss = 1 - GIoU
    
    # Re-calc union properly needed? bbox_iou returns IoU already.
    # We need Union area to calculate GIoU term, but bbox_iou usually returns just iou.
    # Let's assume bbox_iou returns iou. We need internal areas.
    # Duplicating logic for efficiency/correctness:
    
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    area_pred = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    area_target = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
    union = area_pred + area_target - inter_area + eps
    
    iou = inter_area / union
    
    giou = iou - (enc_area - union) / (enc_area + eps)
    loss = 1.0 - giou
    return loss.mean() # or sum

class SimOTAAssigner:
    """
    Simplified SimOTA Assigner.
    Assigns GT to Anchors based on Cost (Cls + Reg).
    """
    def __init__(self, topk=10, cls_weight=1.0, iou_weight=3.0):
        self.topk = topk
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        
    def assign(self, pred_scores, pred_bboxes, anchors, gt_bboxes, gt_labels):
        """
        pred_scores: (N, NumClasses)
        pred_bboxes: (N, 4) - decoded
        anchors: (N, 2 or 4) - centers
        gt_bboxes: (M, 4)
        gt_labels: (M,)
        """
        # 1. Filter anchors inside GT boxes (Basic Prior)
        # 2. Calculate Cost Matrix
        # Cost = L_cls + L_iou
        
        num_gt = gt_bboxes.size(0)
        num_anchors = anchors.size(0)
        
        if num_gt == 0 or num_anchors == 0:
            return torch.full((num_anchors,), -1, dtype=torch.long, device=pred_scores.device)
            
        # Pairwise IoU
        # (N, M)
        # We need a batched IoU function or loop. 
        # For simplicity, using loop or ensuring broadcast.
        # Let's use a simplified center-distance check first to reduce computation?
        # SimOTA uses all valid anchors.
        
        # Calculate IoU (N, M)
        # Using a simplistic broadcasting iou
        # Expand dims
        b_gt = gt_bboxes.unsqueeze(0) # (1, M, 4)
        b_pred = pred_bboxes.unsqueeze(1) # (N, 1, 4)
        
        lt = torch.max(b_gt[..., :2], b_pred[..., :2])
        rb = torch.min(b_gt[..., 2:], b_pred[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        area_gt = (b_gt[..., 2] - b_gt[..., 0]) * (b_gt[..., 3] - b_gt[..., 1])
        area_pred = (b_pred[..., 2] - b_pred[..., 0]) * (b_pred[..., 3] - b_pred[..., 1])
        union = area_gt + area_pred - inter
        iou = inter / (union + 1e-7) # (N, M)
        
        # Cost Calculation
        # Cls Cost: Focal Loss cost (neg log prob)
        # pred_scores (N, C)
        # gt_labels (M)
        # We want score for the specific GT label.
        gt_scores = pred_scores[:, gt_labels] # (N, M)
        cost_cls = F.binary_cross_entropy_with_logits(gt_scores, torch.ones_like(gt_scores), reduction='none')
        # Simplified: -log(score). 
        
        cost_iou = -torch.log(iou.clamp(min=1e-6))
        
        cost = self.cls_weight * cost_cls + self.iou_weight * cost_iou + 10000.0 * (1 - (iou > 0).float()) # Penalize 0 iou
        
        # Dynamic K selection
        # For each GT, select TopK anchors with least cost
        # Actually SimOTA calculates k based on sum of IoU.
        
        dynamic_ks = iou.sum(dim=0).int().clamp(min=1) # (M,)
        
        assigned_gt_inds = torch.full((num_anchors,), -1, dtype=torch.long, device=pred_scores.device)
        
        for gt_idx in range(num_gt):
            k = min(dynamic_ks[gt_idx].item(), self.topk)
            _, topk_inds = torch.topk(cost[:, gt_idx], k, largest=False)
            assigned_gt_inds[topk_inds] = gt_idx
            
        return assigned_gt_inds
