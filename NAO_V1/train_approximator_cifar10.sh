MODEL=cifar10_full
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data
TRIAL_ID=5
SPLIT_TYPE=val_only
TARGET_TYPE=logits
EPOCHS=1
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python approximator.py --data=$DATA_DIR --output_dir=$OUTPUT_DIR --train --evaluate \
    --split_type=$SPLIT_TYPE --epochs=$EPOCHS --target_type=$TARGET_TYPE --trialID=$TRIAL_ID | tee -a $OUTPUT_DIR/approximator_train_$TRIAL_ID.log