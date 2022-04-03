MODEL=cifar10_full
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=2 python arch_embeddings.py --data=$DATA_DIR --output_dir=$OUTPUT_DIR 