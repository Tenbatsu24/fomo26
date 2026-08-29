python3 main.py --config configs/task_1.json --fold 0 || true
# python3 main.py --config configs/task_3.json --fold 0 || true
python3 main.py --config configs/task_5.json --fold 0 || true
python3 main.py --config configs/task_1.json --fold 1 || true
# python3 main.py --config configs/task_3.json --fold 1 || true
python3 main.py --config configs/task_5.json --fold 1 || true
python3 main.py --config configs/task_1.json --fold 2 || true
# python3 main.py --config configs/task_3.json --fold 2 || true
python3 main.py --config configs/task_5.json --fold 2 || true
python3 main.py --config configs/task_1.json --fold 3 || true
# python3 main.py --config configs/task_3.json --fold 3 || true
python3 main.py --config configs/task_5.json --fold 3 || true
python3 main.py --config configs/task_1.json --fold 4 || true
python3 main.py --config configs/task_3.json --fold 4 || true
python3 main.py --config configs/task_5.json --fold 4 || true


nnUNetv2_train 1 3d_fullres 0 -tr UNetViT3DSmallTrainer -p nnUNetViT3DAdaption || true
nnUNetv2_train 2 3d_fullres 0 -tr UNetViT3DSmallTrainer -p nnUNetViT3DAdaption || true
