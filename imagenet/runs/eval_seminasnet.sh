cd ../
export PYTHONPATH=.:$PYTHONPATH

CHECKPOINT=checkpoints/checkpoint.pt
DATA_DIR=data/imagenet/raw-data

python eval.py \
  --data_path=$DATA_DIR \
  --path=$CHECKPOINT
