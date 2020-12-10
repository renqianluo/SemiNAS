cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=search_imagenet
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data

mkdir -p $OUTPUT_DIR

python train_search_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --num_workers=24 \
  --lazy_load \
  --batch_size=512 \
  --eval_batch_size=512 \
  --max_num_updates=20000 \
  --lr=0.4 | tee -a $OUTPUT_DIR/train.log
