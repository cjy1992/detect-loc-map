CUDA_VISIBLE_DEVICES=1 python train_eval.py \
  --root_dir logs \
  --experiment_name 128_128_50k_focal_concat_loc_norecons \
  --gin_file params.gin \
  --gin_param load_carla_env.port=3000 \
  --gin_param train_eval.model_batch_size=1 \
  --gin_param train_eval.obs_size=128 \
  --gin_param train_eval.pixor_size=128 \
  --gin_param train_eval.sequence_length=10 
