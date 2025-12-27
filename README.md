  
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python train_det.py --data_root "d:\ml-workspace\RTMDet-RTMPose\custom_dataset" --batch_size 8 --epochs 10

python train_pose.py --data_root "d:\ml-workspace\RTMDet-RTMPose\custom_dataset" --batch_size 8 --epochs 10

python train_det.py --data_root custom_dataset --epochs 100 --batch_size 8

python train_pose.py --data_root custom_dataset --epochs 100 --batch_size 8

python train_det.py --data_root custom_dataset --epochs 100 --batch_size 8 --resume train/weights/rtmdet_custom.pth

python train_pose.py --data_root custom_dataset --epochs 100 --batch_size 8 --resume train/weights/rtmpose_custom.pth


venv/bin/python train_pose.py --data_root custom_dataset --epochs 50 --batch_size 8

venv/bin/python train_pose.py --data_root custom_dataset --epochs 150 --batch_size 8 --resume train/weights/rtmpose_custom.pth


