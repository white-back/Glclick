MODEL_PATH=/data2/user/2023/cbj/SimpleClick-final/experiments/iter_mask/cocolvis_plainvit_base448/007/checkpoints/189.pth

python scripts/evaluate_model.py NoBRS \
--gpu=1 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--target-iou=0.95 \
--datasets=Berkeley,DAVIS,COCO_MVal,SBD

