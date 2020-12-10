cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=SemiNASNet
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data
ARCH="6 5 1 7 4 2 5 3 5 2 5 3 3 5 7 3 1 6 3 3 6"

mkdir -p $OUTPUT_DIR

python train_imagenet.py \
  --data_path=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --lazy_load \
  --arch="$ARCH" \
  --dropout=0.3 \
  | tee -a $OUTPUT_DIR/train.log
