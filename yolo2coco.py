
import json
import os
import glob
import struct

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external dependencies
    """
    with open(file_path, 'rb') as f:
        head = f.read(24)
        if len(head) != 24:
            return None
        
        if head.startswith(b'\211PNG\r\n\032\n'):
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return None
            width, height = struct.unpack('>ii', head[16:24])
            return width, height
            
        elif head.startswith(b'\xff\xd8'):
            try:
                f.seek(0)
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                    f.seek(size, 1)
                    byte = f.read(1)
                    while ord(byte) == 0xff:
                        byte = f.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', f.read(2))[0] - 2
                
                # We are at a SOFn block
                f.seek(1, 1)  # precision
                h, w = struct.unpack('>HH', f.read(4))
                return w, h
            except Exception:
                return None
    return None

def yolo_to_coco(source_dir, output_dir, split):
    """
    Converts YOLO format dataset to COCO format JSON.
    Handles standard YOLO detection (5 values) and YOLO-Pose (5 + 3*kpt values).
    """
    
    images_dir = os.path.join(source_dir, "images", split)
    labels_dir = os.path.join(source_dir, "labels", split)
    
    # Check if directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Skipping {split}: Directory not found.")
        return

    # COCO Categories (Assuming single class 'person')
    categories = [
        {
            "id": 1, 
            "name": "person", 
            "supercategory": "person",
            "keypoints": [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ],
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            ]
        }
    ]
    
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    img_id_map = {}
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(images_dir, "*.png")) + \
                  glob.glob(os.path.join(images_dir, "*.jpeg"))
    
    image_files.sort()
    
    ann_id = 1
    
    print(f"Processing {split} set...")
    total_files = len(image_files)
    
    for img_idx, img_path in enumerate(image_files):
        if img_idx % 100 == 0:
            print(f"Propcessing {img_idx}/{total_files}...", end="\r")
            
        filename = os.path.basename(img_path)
        img_id = img_idx + 1 # 1-based index
        img_id_map[filename] = img_id
        
        try:
            dims = get_image_size(img_path)
            if dims is None:
                print(f"Warning: Could not determine size for {img_path}")
                continue
            width, height = dims
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue
            
        coco_dataset["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        # Determine corresponding label file
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 0:
                    continue
                
                cls_id = int(parts[0])
                # Remap class 0 to 1 if we only have one class. 
                # If dataset mentions specific classes, we'd adjust here.
                # Assuming class 0 is 'person'.
                category_id = 1 
                
                # YOLO format: x_center, y_center, w, h (normalized)
                x_c, y_c, w, h = parts[1:5]
                
                # Convert to COCO format: x_min, y_min, w, h (absolute)
                abs_w = w * width
                abs_h = h * height
                abs_x_min = (x_c * width) - (abs_w / 2)
                abs_y_min = (y_c * height) - (abs_h / 2)
                
                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [abs_x_min, abs_y_min, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                }
                
                # Parse Keypoints if available (YOLO-Pose shape: 5 + 3*k)
                if len(parts) > 5:
                    keypoints_raw = parts[5:]
                    keypoints = []
                    num_keypoints = 0
                    for i in range(0, len(keypoints_raw), 3):
                        # Ensure we don't go out of bounds if line is malformed
                        if i+2 >= len(keypoints_raw):
                            break
                            
                        kx = keypoints_raw[i] * width
                        ky = keypoints_raw[i+1] * height
                        kv = keypoints_raw[i+2] # Visibility or conf
                        
                        # Convert YOLO visibility/conf to COCO visibility (0, 1, 2)
                        # v=0: not labeled, v=1: labeled not visible, v=2: labeled visible
                        if kv < 0.01:
                            v = 0
                        else:
                            v = 2
                            num_keypoints += 1
                        
                        keypoints.extend([kx, ky, v])
                        
                    annotation["keypoints"] = keypoints
                    annotation["num_keypoints"] = num_keypoints
                
                coco_dataset["annotations"].append(annotation)
                ann_id += 1
    
    print(f"Processed {total_files} images.")
    output_json = os.path.join(output_dir, f"instances_{split}.json")
    
    with open(output_json, "w") as f:
        json.dump(coco_dataset, f)
    
    print(f"Saved {output_json} with {len(coco_dataset['images'])} images and {len(coco_dataset['annotations'])} annotations.")

if __name__ == "__main__":
    base_dir = "custom_dataset_6000"
    output_base = os.path.join(base_dir, "annotations")
    os.makedirs(output_base, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        yolo_to_coco(base_dir, output_base, split)
