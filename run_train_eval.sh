CUDA_VISIBLE_DEVICES=0 python train_eval.py \
  --root_dir logs \
  --experiment_name e2e_perception \
  --gin_file params.gin \
  --gin_param load_carla_env.port=2000 \
  --gin_param train_eval.model_batch_size=32 \
  --gin_param train_eval.obs_size=128 \
  --gin_param train_eval.pixor_size=128 \
  --gin_param train_eval.sequence_length=10 
  
