cd ..
export PYTHONPATH=.:$PYTHONPATH
MODEL=seminas
OUTPUT_DIR=outputs/$MODEL

mkdir -p $OUTPUT_DIR

python train_seminas.py \
  --output_dir=$OUTPUT_DIR \
  | tee $OUTPUT_DIR/log.txt
