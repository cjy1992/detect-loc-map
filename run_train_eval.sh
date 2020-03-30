CUDA_VISIBLE_DEVICES=2 python train_eval.py \
  --root_dir logs \
  --experiment_name multi_camera \
  --gin_file params.gin \
  --gin_param load_carla_env.port=4000 \
  --gin_param train_eval.model_batch_size=32 \
  --gin_param train_eval.obs_size=128 \
  --gin_param train_eval.pixor_size=128 \
  --gin_param train_eval.sequence_length=7 \
  --gin_param train_eval.num_eval_episodes=1
