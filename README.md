  
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python train_det.py --data_root "d:\ml-workspace\RTMDet-RTMPose\custom_dataset" --batch_size 8 --epochs 10

python train_pose.py --data_root "d:\ml-workspace\RTMDet-RTMPose\custom_dataset" --batch_size 8 --epochs 10

python train_det.py --data_root custom_dataset --epochs 100 --batch_size 8

python train_pose.py --data_root custom_dataset --epochs 100 --batch_size 8

python train_det.py --data_root custom_dataset --epochs 100 --batch_size 8 --resume train/weights/rtmdet_custom.pth

python train_pose.py --data_root custom_dataset --epochs 100 --batch_size 8 --resume train/weights/rtmpose_custom.pth