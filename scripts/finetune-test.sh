# this script finetunes a model on a test dataset

# python \
#     src/train/NetfoundFinetuning.py \
#     --train_dir /mnt/extra/data/iot2023/small-2 \
#     --model_name_or_path /mnt/extra/models/netFound-640M-base \
#     --output_dir /mnt/extra/models/cic-iot-2023 \
#     --report_to tensorboard \
#     --overwrite_output_dir \
#     --save_safetensors false \
#     --do_train \
#     --do_eval \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --learning_rate 0.001 \
#     --num_train_epochs 1 \
#     --problem_type single_label_classification \
#     --num_labels 34 \
#     --load_best_model_at_end \
#     --netfound_large True \
#     --freeze_base True \
#     --validation_split_percentage 20 \
#     --bf16 \
#     --hidden_size 1024 \
#     --num_hidden_layers 24 \
#     --num_attention_heads 16 \
#     --dataloader_num_workers 8 \
#     --per_device_eval_batch_size 40 \
#     --per_device_train_batch_size 40

python \
    src/train/NetfoundFinetuning.py \
    --train_dir /mnt/extra/processed/iot2023/iot2023-8class-big-benign \
    --model_name_or_path /mnt/extra/models/netFound-640M-base \
    --output_dir /mnt/extra/models/iot2023-hr \
    --report_to tensorboard \
    --overwrite_output_dir \
    --save_safetensors false \
    --do_feature_extraction \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate 0.0002 \
    --num_train_epochs 1 \
    --problem_type single_label_classification \
    --num_labels 8 \
    --load_best_model_at_end \
    --netfound_large True \
    --freeze_base True \
    --validation_split_percentage 20 \
    --bf16 \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --n_estimators 100
    # --per_device_train_batch_size 32 \
    # --data_cache_dir /mnt/extra/tmp

