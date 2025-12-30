  
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt



python train_det.py --data_root custom_dataset_6000 --batch_size 4 --epochs 50 --device mps

python train_pose.py --data_root custom_dataset_6000 --batch_size 8 --epochs 50

python train_det.py --data_root custom_dataset_6000 --batch_size 4 --epochs 12 --device mps --resume train/weights/rtmdet_custom.pth

python train_det.py --config configs/rtmdet.yaml --device cuda