CUDA_VISIBLE_DEVICES=2 python perception_driving/agents/perception/examples/v2/train_eval.py \
  --root_dir logs \
  --experiment_name 128_128_50k_focal_concat_loc_norecons \
  --gin_file params.gin \
  --gin_param load_carla_env.port=4000 \
  --gin_param train_eval.model_batch_size=32 \
  --gin_param train_eval.obs_size=128 \
  --gin_param train_eval.pixor_size=128 \
  --gin_param train_eval.sequence_length=10 
