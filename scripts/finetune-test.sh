# this script finetunes a model on a test dataset

# python \
#     src/train/NetfoundFinetuning.py \
#     --train_dir /mnt/extra/processed/iot2023/iot2023-8class-http \
#     --model_name_or_path /mnt/extra/models/netFound-640M-base \
#     --output_dir /mnt/extra/models/iot2023-12layer \
#     --report_to tensorboard \
#     --overwrite_output_dir \
#     --save_safetensors false \
#     --do_train_feature_extractor \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --learning_rate 1e-5 \
#     --num_train_epochs 1 \
#     --problem_type single_label_classification \
#     --num_labels 8 \
#     --load_best_model_at_end \
#     --netfound_large True \
#     --freeze_base True \
#     --validation_split_percentage 20 \
#     --bf16 \
#     --dataloader_num_workers 8 \
#     --per_device_eval_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 10 \
#     --layers_to_unfreeze 12 \

python \
    src/train/NetfoundFinetuning.py \
    --train_dir /mnt/extra/processed/iot2023/iot2023-8class-http \
    --model_name_or_path /mnt/extra/models/netFound-640M-base \
    --finetuned_base_dir /mnt/extra/models/iot2023-12layer \
    --ensemble_rf_dir /mnt/extra/models/iot2023-6layer-rf \
    --hr_dir /mnt/extra/models/iot2023-hr-ft-base \
    --output_dir /mnt/extra/models/iot2023-ensemble-12-layer-and-rf \
    --report_to tensorboard \
    --overwrite_output_dir \
    --save_safetensors false \
    --do_ensemble \
    --eval_strategy epoch \
    --save_strategy epoch \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --problem_type single_label_classification \
    --num_labels 8 \
    --load_best_model_at_end \
    --netfound_large True \
    --freeze_base True \
    --validation_split_percentage 20 \
    --bf16 \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 10 \
    --layers_to_unfreeze 12 \
