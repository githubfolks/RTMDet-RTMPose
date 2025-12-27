"""
Analyze training data quality for pose estimation.
Checks for potential issues in annotations.
"""
import os
import glob
import numpy as np
from collections import defaultdict

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def parse_label_file(filepath):
    """Parse YOLO format label file with keypoints."""
    annotations = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # Parse keypoints (x, y, visibility) * 17
            keypoints = []
            kpt_data = parts[5:]
            for i in range(0, len(kpt_data), 3):
                if i + 2 < len(kpt_data):
                    kx = float(kpt_data[i])
                    ky = float(kpt_data[i + 1])
                    vis = int(float(kpt_data[i + 2]))
                    keypoints.append((kx, ky, vis))
            
            annotations.append({
                'class': cls_id,
                'bbox': (cx, cy, w, h),
                'keypoints': keypoints
            })
    return annotations

def analyze_dataset(label_dir):
    """Analyze annotation quality."""
    stats = {
        'total_files': 0,
        'total_persons': 0,
        'persons_per_image': [],
        'bbox_sizes': [],
        'keypoint_visibility': defaultdict(lambda: {'visible': 0, 'occluded': 0, 'not_labeled': 0}),
        'small_bboxes': 0,
        'edge_cases': 0,
        'missing_keypoints': defaultdict(int),
        'keypoint_in_bbox': defaultdict(lambda: {'inside': 0, 'outside': 0}),
        'symmetric_issues': [],
        'anatomically_incorrect': [],
    }
    
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    stats['total_files'] = len(label_files)
    
    for lf in label_files:
        annotations = parse_label_file(lf)
        stats['persons_per_image'].append(len(annotations))
        
        for ann in annotations:
            stats['total_persons'] += 1
            cx, cy, w, h = ann['bbox']
            stats['bbox_sizes'].append((w, h))
            
            # Check for small bounding boxes
            if w < 0.05 or h < 0.1:
                stats['small_bboxes'] += 1
            
            # Check for edge cases (person at image edge)
            if cx - w/2 < 0.02 or cx + w/2 > 0.98 or cy - h/2 < 0.02 or cy + h/2 > 0.98:
                stats['edge_cases'] += 1
            
            keypoints = ann['keypoints']
            
            # Check keypoint visibility
            for i, (kx, ky, vis) in enumerate(keypoints):
                if i >= len(KEYPOINT_NAMES):
                    break
                    
                if vis == 2:
                    stats['keypoint_visibility'][i]['visible'] += 1
                elif vis == 1:
                    stats['keypoint_visibility'][i]['occluded'] += 1
                else:
                    stats['keypoint_visibility'][i]['not_labeled'] += 1
                    stats['missing_keypoints'][i] += 1
                
                # Check if keypoint is within bbox
                x1 = cx - w/2
                x2 = cx + w/2
                y1 = cy - h/2
                y2 = cy + h/2
                
                # Allow 20% margin outside bbox
                margin = 0.2
                x1_m = x1 - w * margin
                x2_m = x2 + w * margin
                y1_m = y1 - h * margin
                y2_m = y2 + h * margin
                
                if x1_m <= kx <= x2_m and y1_m <= ky <= y2_m:
                    stats['keypoint_in_bbox'][i]['inside'] += 1
                else:
                    stats['keypoint_in_bbox'][i]['outside'] += 1
            
            # Check for anatomical issues
            if len(keypoints) >= 17:
                # Shoulders should be at similar height
                l_shoulder = keypoints[5]
                r_shoulder = keypoints[6]
                if l_shoulder[2] == 2 and r_shoulder[2] == 2:
                    if abs(l_shoulder[1] - r_shoulder[1]) > h * 0.3:
                        stats['anatomically_incorrect'].append(('shoulder_height', os.path.basename(lf)))
                
                # Hips should be at similar height
                l_hip = keypoints[11]
                r_hip = keypoints[12]
                if l_hip[2] == 2 and r_hip[2] == 2:
                    if abs(l_hip[1] - r_hip[1]) > h * 0.2:
                        stats['anatomically_incorrect'].append(('hip_height', os.path.basename(lf)))
                
                # Left/right eye swap check
                l_eye = keypoints[1]
                r_eye = keypoints[2]
                nose = keypoints[0]
                if l_eye[2] == 2 and r_eye[2] == 2 and nose[2] == 2:
                    # In a front-facing view, left_eye should be to the right of right_eye
                    # This can vary based on view angle, so we check for extreme cases
                    pass
    
    return stats

def print_report(stats):
    """Print analysis report."""
    print("=" * 70)
    print("TRAINING DATA QUALITY ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 50)
    print(f"  Total label files: {stats['total_files']}")
    print(f"  Total persons annotated: {stats['total_persons']}")
    avg_persons = np.mean(stats['persons_per_image']) if stats['persons_per_image'] else 0
    print(f"  Average persons per image: {avg_persons:.2f}")
    print(f"  Max persons in single image: {max(stats['persons_per_image']) if stats['persons_per_image'] else 0}")
    
    print(f"\n📦 BOUNDING BOX ANALYSIS")
    print("-" * 50)
    if stats['bbox_sizes']:
        widths = [s[0] for s in stats['bbox_sizes']]
        heights = [s[1] for s in stats['bbox_sizes']]
        print(f"  Width  - Min: {min(widths):.4f}, Max: {max(widths):.4f}, Mean: {np.mean(widths):.4f}")
        print(f"  Height - Min: {min(heights):.4f}, Max: {max(heights):.4f}, Mean: {np.mean(heights):.4f}")
    print(f"  ⚠️  Small bboxes (w<0.05 or h<0.1): {stats['small_bboxes']} ({100*stats['small_bboxes']/max(1,stats['total_persons']):.1f}%)")
    print(f"  ⚠️  Edge cases (person at image edge): {stats['edge_cases']} ({100*stats['edge_cases']/max(1,stats['total_persons']):.1f}%)")
    
    print(f"\n🦴 KEYPOINT VISIBILITY ANALYSIS")
    print("-" * 50)
    print(f"  {'Keypoint':<15} {'Visible':>10} {'Occluded':>10} {'Missing':>10} {'Outside BBox':>12}")
    print("  " + "-" * 57)
    
    issues_found = []
    for i in range(17):
        vis = stats['keypoint_visibility'][i]
        total = vis['visible'] + vis['occluded'] + vis['not_labeled']
        vis_pct = 100 * vis['visible'] / max(1, total)
        
        outside = stats['keypoint_in_bbox'][i]['outside']
        outside_pct = 100 * outside / max(1, total)
        
        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kpt_{i}"
        print(f"  {name:<15} {vis['visible']:>10} {vis['occluded']:>10} {vis['not_labeled']:>10} {outside:>10} ({outside_pct:.1f}%)")
        
        if vis_pct < 70:
            issues_found.append(f"Low visibility rate for {name}: {vis_pct:.1f}%")
        if outside_pct > 5:
            issues_found.append(f"High outside-bbox rate for {name}: {outside_pct:.1f}%")
    
    print(f"\n🔍 ANATOMICAL CONSISTENCY CHECK")
    print("-" * 50)
    if stats['anatomically_incorrect']:
        issue_types = defaultdict(int)
        for issue, filename in stats['anatomically_incorrect']:
            issue_types[issue] += 1
        for issue, count in issue_types.items():
            print(f"  ⚠️  {issue}: {count} instances")
    else:
        print("  ✅ No obvious anatomical issues detected")
    
    print(f"\n📋 POTENTIAL ISSUES SUMMARY")
    print("-" * 50)
    
    total_issues = 0
    
    # Check for keypoints outside bbox
    outside_total = sum(stats['keypoint_in_bbox'][i]['outside'] for i in range(17))
    if outside_total > 0:
        print(f"  ⚠️  {outside_total} keypoints found outside bounding boxes")
        total_issues += 1
    
    # Check visibility rates
    low_vis_kpts = []
    for i in range(17):
        vis = stats['keypoint_visibility'][i]
        total = vis['visible'] + vis['occluded'] + vis['not_labeled']
        if total > 0 and vis['visible'] / total < 0.8:
            low_vis_kpts.append(KEYPOINT_NAMES[i])
    if low_vis_kpts:
        print(f"  ⚠️  Low visibility rate (<80%) for: {', '.join(low_vis_kpts)}")
        total_issues += 1
    
    if stats['small_bboxes'] > stats['total_persons'] * 0.1:
        print(f"  ⚠️  High proportion of small bboxes: {100*stats['small_bboxes']/stats['total_persons']:.1f}%")
        total_issues += 1
    
    if stats['edge_cases'] > stats['total_persons'] * 0.3:
        print(f"  ⚠️  High proportion of edge cases: {100*stats['edge_cases']/stats['total_persons']:.1f}%")
        total_issues += 1
    
    if total_issues == 0:
        print("  ✅ No major data quality issues detected")
    
    print("\n" + "=" * 70)
    
    return total_issues

def main():
    train_label_dir = 'custom_dataset/labels/train'
    val_label_dir = 'custom_dataset/labels/val'
    
    print("\n🔬 Analyzing TRAINING data...")
    train_stats = analyze_dataset(train_label_dir)
    train_issues = print_report(train_stats)
    
    print("\n\n🔬 Analyzing VALIDATION data...")
    val_stats = analyze_dataset(val_label_dir)
    val_issues = print_report(val_stats)
    
    print("\n\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    
    total_issues = train_issues + val_issues
    if total_issues == 0:
        print("✅ Data quality looks good. Consider:")
        print("   - Increasing training epochs (200-300)")
        print("   - Adding more data augmentation")
        print("   - Fine-tuning learning rate")
    elif total_issues <= 2:
        print("⚠️  Minor data issues detected. Consider:")
        print("   - Reviewing annotations for flagged keypoints")
        print("   - Adding data augmentation to handle edge cases")
    else:
        print("❌ Multiple data quality issues detected. Recommended actions:")
        print("   - Review and fix annotations for problem areas")
        print("   - Consider re-annotating problematic samples")
        print("   - Filter out low-quality annotations before training")

if __name__ == "__main__":
    main()
