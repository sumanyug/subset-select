MODEL=search_cifar10_debug
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1 python train_search.py --data=$DATA_DIR --output_dir=$OUTPUT_DIR --debug | tee -a $OUTPUT_DIR/train.log
