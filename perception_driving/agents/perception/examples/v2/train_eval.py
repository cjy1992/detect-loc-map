from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from shapely.geometry import Polygon

import functools
import gin
import numpy as np
import os
import tensorflow as tf
import time
import collections
import cv2

import gym
import gym_carla

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import q_rnn_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from interp_e2e_driving.environments import filter_observation_wrapper
from interp_e2e_driving.utils import gif_utils

from perception_driving.agents.perception import perception_agent
from perception_driving.networks import sequential_latent_pixor_network
from perception_driving.networks import state_based_heuristic_actor_network


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name used for naming the output directory.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS

@gin.configurable
def load_carla_env(
  env_name='carla-v0',
  discount=1.0,
  number_of_vehicles=100,
  number_of_walkers=0,
  display_size=256,
  max_past_step=1,
  dt=0.1,
  discrete=False,
  discrete_acc=[-3.0, 0.0, 3.0],
  discrete_steer=[-0.2, 0.0, 0.2],
  continuous_accel_range=[-3.0, 3.0],
  continuous_steer_range=[-0.3, 0.3],
  ego_vehicle_filter='vehicle.lincoln*',
  port=2000,
  town='Town03',
  task_mode='random',
  max_time_episode=500,
  max_waypt=12,
  obs_range=32,
  lidar_bin=0.5,
  d_behind=12,
  out_lane_thres=2.7,
  desired_speed=8,
  max_ego_spawn_times=200,
  target_waypt_index=5,
  display_route=True,
  pixor_size=64,
  obs_channels=None,
  action_repeat=1):
  """Loads train and eval environments."""
  env_params = {
    'number_of_vehicles': number_of_vehicles,
    'number_of_walkers': number_of_walkers,
    'display_size': display_size,  # screen size of bird-eye render
    'max_past_step': max_past_step,  # the number of past steps to draw
    'dt': dt,  # time interval between two frames
    'discrete': discrete,  # whether to use discrete control space
    'discrete_acc': discrete_acc,  # discrete value of accelerations
    'discrete_steer': discrete_steer,  # discrete value of steering angles
    'continuous_accel_range': continuous_accel_range,  # continuous acceleration range
    'continuous_steer_range': continuous_steer_range,  # continuous steering angle range
    'ego_vehicle_filter': ego_vehicle_filter,  # filter for defining ego vehicle
    'port': port,  # connection port
    'town': town,  # which town to simulate
    'task_mode': task_mode,  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': max_time_episode,  # maximum timesteps per episode
    'max_waypt': max_waypt,  # maximum number of waypoints
    'obs_range': obs_range,  # observation range (meter)
    'lidar_bin': lidar_bin,  # bin size of lidar sensor (meter)
    'd_behind': d_behind,  # distance behind the ego vehicle (meter)
    'out_lane_thres': out_lane_thres,  # threshold for out of lane
    'desired_speed': desired_speed,  # desired speed (m/s)
    'max_ego_spawn_times': max_ego_spawn_times,  # maximum times to spawn ego vehicle
    'target_waypt_index': target_waypt_index,  # index of the target way point
    'display_route': display_route,  # whether to render the desired route
    'pixor_size': pixor_size,  # size of the pixor labels
  }

  gym_spec = gym.spec(env_name)
  gym_env = gym_spec.make(params=env_params)

  if obs_channels:
    gym_env = filter_observation_wrapper.FilterObservationWrapper(gym_env, obs_channels)

  py_env = gym_wrapper.GymWrapper(
    gym_env,
    discount=discount,
    auto_reset=True,
  )

  eval_py_env = py_env

  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)

  return py_env, eval_py_env


def compute_summaries(metrics,
                      environment,
                      policy,
                      train_step=None,
                      summary_writer=None,
                      num_episodes=1,
                      num_episodes_to_render=1,
                      model_net=None,
                      fps=10,
                      image_keys=None,
                      pixor_size=128):
  for metric in metrics:
    metric.reset()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  if num_episodes_to_render:
    images = [[time_step.observation]]  # now images contain dictionary of images
    latents = [[policy_state[1]]]
  else:
    images = []
    latents = []

  episode = 0
  while episode < num_episodes:
    action_step = policy.action(time_step, policy_state)
    next_time_step = environment.step(action_step.action)
    policy_state = action_step.state

    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    for observer in metrics:
      observer(traj)

    if episode < num_episodes_to_render:
      if traj.is_boundary():
        images.append([])
        latents.append([])
      images[-1].append(next_time_step.observation)
      latents[-1].append(policy_state[1])

    if traj.is_last():
      episode += 1
      policy_state = policy.get_initial_state(environment.batch_size)

    time_step = next_time_step

  if train_step and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = m.name
        tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)

  # shape of images is [[images in episode as timesteps]]
  if images:
    if type(images[0][0]) is collections.OrderedDict:
      videos = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=True)
      videos_bb = pad_and_concatenate_bb_videos(images, is_dict=True, pixor_size=pixor_size)
    else:
      videos = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=False)
      videos_bb = pad_and_concatenate_bb_videos(images, is_dict=False, pixor_size=pixor_size)
    videos = tf.image.convert_image_dtype([videos], tf.uint8, saturate=True)
    videos = tf.squeeze(videos, axis=2)
    videos_bb = tf.image.convert_image_dtype([videos_bb], tf.uint8, saturate=True)
    videos_bb = tf.squeeze(videos_bb, axis=2)

    reconstruct_images = get_latent_reconstruction_videos(latents, model_net, pixor_size=pixor_size)
    reconstruct_images = tf.image.convert_image_dtype([reconstruct_images], tf.uint8, saturate=True)
    reconstruct_images = tf.squeeze(reconstruct_images, axis=2)

    reconstruct_bb_images = get_latent_reconstruction_bb_videos(images, latents, model_net, pixor_size=pixor_size)
    reconstruct_bb_images = tf.image.convert_image_dtype([reconstruct_bb_images], tf.uint8, saturate=True)
    reconstruct_bb_images = tf.squeeze(reconstruct_bb_images, axis=2)

    # Comment if obs_size != pixor_size
    # videos = tf.concat([videos, videos_bb], axis=-2)
    # reconstruct_images = tf.concat([reconstruct_images, reconstruct_bb_images], axis=-2)

    # Need to avoid eager here to avoid rasing error
    gif_summary = common.function(gif_utils.gif_summary_v2)

    gif_summary('ObservationVideoEvalPolicy', videos, 1, fps)
    gif_summary('ReconstructedVideoEvalPolicy', reconstruct_images, 1, fps)

    bb = tf.concat([videos_bb, reconstruct_bb_images], axis=-2)
    gif_summary('bbVideoEvalPolicy', bb, 1, fps)    

    # gif_summary('ObservationBBVideoEvalPolicy', videos_bb, 1, fps)
    # gif_summary('ReconstructedVideoBBEvalPolicy', reconstruct_bb_images, 1, fps)


def get_eval_metrics(images, latents, model_net):
  for i in range(len(latents)):
    latent_eps = latents[i]
    image_eps = images[i]
    for j in range(len(latent_eps)):
      latent = latent_eps[j]
      image_dict = image_eps[j]
      


def pad_and_concatenate_videos(videos, image_keys, is_dict=False):
  max_episode_length = max([len(video) for video in videos])
  if is_dict:
    # videos = [[tf.concat(list(dict_obs.values()), axis=2) for dict_obs in video] for video in videos]
    videos = [[tf.concat([dict_obs[key] for key in image_keys], axis=2) for dict_obs in video] for video in videos]
  
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=1) for frames in zip(*videos)]
  return videos


def pad_and_concatenate_bb_videos(videos, is_dict=False, pixor_size=128):
  max_episode_length = max([len(video) for video in videos])
  if is_dict:
    # videos = [[tf.concat(list(dict_obs.values()), axis=2) for dict_obs in video] for video in videos]
    # tf.concat([dict_obs[key] for key in image_keys], axis=2) 
    videos = [[get_bb_bev_from_obs(dict_obs, pixor_size) for dict_obs in video] for video in videos]
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=1) for frames in zip(*videos)]
  # print(videos)
  return videos


# def get_concat_gt(dict_obs, pixor_size):
#   image_bb = get_bb_bev_from_obs(dict_obs, pixor_size)
#   image = tf.concat([dict_obs['camera'], dict_obs['lidar'], dict_obs['roadmap'], image_bb], axis=-2)
#   return image


def get_latent_reconstruction_videos(latents, model_net, pixor_size=128):
  videos = []
  for latent_eps in latents:
    videos.append([])
    for latent in latent_eps:
      image = model_net.reconstruct(latent)
      # width = image.shape[-2]
      # camera = tf.gather(image, tf.range(0,width/3), axis=-2)
      # lidar = tf.gather(image, tf.range(width/3,width/3*2), axis=-2)
      # roadmap = tf.gather(image, tf.range(width/3*2,width), axis=-2)
      # dict_recons = model_net.reconstruct_pixor(latent)
      # dict_recons.update({
      #   'camera':camera,
      #   'lidar':lidar,
      #   'roadmap':roadmap,
      #   })
      # image_bb = get_bb_bev_from_obs(dict_recons, pixor_size)  # (B,H,W,3)
      # image_bb = tf.image.convert_image_dtype(image_bb, dtype=tf.float32)
      # image = tf.concat([image, image_bb], axis=-2)
      # image = tf.concat([dict_recons['camera'], image_bb, dict_recons['roadmap']], axis=-2)
      videos[-1].append(image)

  max_episode_length = max([len(video) for video in videos])
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=1) for frames in zip(*videos)]
  return videos


def get_latent_reconstruction_bb_videos(images, latents, model_net, pixor_size=128):
  videos = []
  for i in range(len(latents)):
    latent_eps = latents[i]
    videos.append([])
    for j in range(len(latent_eps)):
      latent = latent_eps[j]
      lidar = images[i][j]['lidar']
      dict_recons = model_net.reconstruct_pixor(latent)
      dict_recons.update({
        'lidar':lidar,
        })
      image_bb = get_bb_bev_from_obs(dict_recons, pixor_size)  # (B,H,W,3)
      vh_clas = tf.image.convert_image_dtype(dict_recons['vh_clas'], dtype=tf.uint8)
      tile_shape = [1]*len(vh_clas.shape)
      tile_shape[-1] = 3
      image_clas = tf.tile(vh_clas, tile_shape)  # (B,H,W,3)
      image = tf.concat([image_clas, image_bb], axis=-2)
      videos[-1].append(image)

  max_episode_length = max([len(video) for video in videos])
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=1) for frames in zip(*videos)]
  return videos


def get_bb_bev_from_obs(dict_obs, pixor_size=128):
  """Input dict_obs with (B,H,W,C), return (B,H,W,3)"""
  vh_clas = tf.squeeze(dict_obs['vh_clas'], axis=-1)  # (B,H,W,1)
  # vh_clas = tf.gather(vh_clas, 0, axis=-1)  # (B,H,W)
  vh_regr = dict_obs['vh_regr']  # (B,H,W,6)
  decoded_reg = decode_reg(vh_regr, pixor_size)  # (B,H,W,8)

  lidar = dict_obs['lidar']

  B = vh_regr.shape[0]
  images = []
  for i in range(B):
    corners, _ = pixor_postprocess(vh_clas[i], decoded_reg[i])  # (N,4,2)
    image = get_bev(lidar, corners, pixor_size)  # (H,W,3)
    images.append(image)
  images = tf.convert_to_tensor(images, dtype=np.uint8)  # (B,H,W,3)
  return images


def decode_reg(vh_regr, pixor_size=128):
  # Tensor in (B, H, W, 1)
  cos_t, sin_t, dx, dy, logw, logl = tf.split(vh_regr, 6, axis=-1)
  yaw = tf.atan2(sin_t, cos_t)
  cos_t = tf.cos(yaw)
  sin_t = tf.sin(yaw)

  B = cos_t.shape[0]

  # Get the pixels positions coordinates, so it's left-handed coordinate
  # and the origin is at left-bottom
  y = tf.range(0, pixor_size, dtype=tf.float32)
  x = tf.range(pixor_size-1, -1, -1, dtype=tf.float32)
  yy, xx = tf.meshgrid(y, x)
  xx = tf.expand_dims(tf.expand_dims(xx, -1), 0)  # (1, H, W, 1)
  yy = tf.expand_dims(tf.expand_dims(yy, -1), 0)
  xx = tf.tile(xx, [B,1,1,1])
  yy = tf.tile(yy, [B,1,1,1])

  # dy, dy is center_pixel-pixel_pos
  centre_y = yy + dy
  centre_x = xx + dx
  l = tf.exp(logl)  # half the length
  w = tf.exp(logw)

  # Shape (B, H, W, 1)
  rear_left_x = centre_x - l * cos_t + w * sin_t
  rear_left_y = centre_y - l * sin_t - w * cos_t
  rear_right_x = centre_x - l * cos_t - w * sin_t
  rear_right_y = centre_y - l * sin_t + w * cos_t
  front_right_x = centre_x + l * cos_t - w * sin_t
  front_right_y = centre_y + l * sin_t + w * cos_t
  front_left_x = centre_x + l * cos_t + w * sin_t
  front_left_y = centre_y + l * sin_t - w * cos_t

  # Shape (B, H, W, 8)
  decoded_reg = tf.concat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                           front_right_x, front_right_y, front_left_x, front_left_y], axis=-1)

  return decoded_reg


def pixor_postprocess(vh_clas, decoded_reg, cls_threshold=0.5):
  """Return bounding-box image with shape (H, W, 3).
    
    vh_clas: (H, W) tensor
    decoded_reg: (H, W, 8) tensor
  """
  activation = vh_clas.numpy() > cls_threshold
  num_boxes = int(activation.sum())

  if num_boxes == 0:  # No bounding boxes
    return [], []

  corners = tf.boolean_mask(decoded_reg, activation)  # (N,8)
  corners = tf.reshape(corners, (-1, 4, 2)).numpy()  # (N,4,2)
  scores = tf.boolean_mask(vh_clas, activation).numpy()  # (N,)

  # NMS
  selected_ids = non_max_suppression(corners, scores)
  corners = corners[selected_ids]
  scores = scores[selected_ids]

  return corners, scores

def get_bev(background, corners, pixor_size=128):
  # intensity = 0*np.ones((pixor_size, pixor_size, 3), dtype=np.uint8) 
  background = np.rot90(background[0].numpy(), 3)
  intensity = cv2.resize(background, (pixor_size,pixor_size), interpolation = cv2.INTER_AREA)
  # intensity = background[0].numpy()

  # FLip in the x direction

  if corners is not None:
    for corners in corners:
      plot_corners = corners.astype(int).reshape((-1, 1, 2))
      cv2.polylines(intensity, [plot_corners], True, (255, 255, 0), 1)
      cv2.line(intensity, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 2)

  image = intensity.astype(np.uint8)
  image = np.rot90(image, 1)

  return image


def non_max_suppression(boxes, scores, threshold=0.1):
  """Performs non-maximum suppression and returns indices of kept boxes.
  boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
  scores: 1-D array of box scores.
  threshold: Float. IoU threshold to use for filtering.
  return an numpy array of the positions of picks
  """
  assert boxes.shape[0] > 0
  if boxes.dtype.kind != "f":
    boxes = boxes.astype(np.float32)

  polygons = convert_format(boxes)

  top = 128*10
  # Get indicies of boxes sorted by scores (highest first)
  ixs = scores.argsort()[::-1][:top]

  pick = []
  while len(ixs) > 0:
    # Pick top box and add its index to the list
    i = ixs[0]
    pick.append(i)
    # Compute IoU of the picked box with the rest
    iou = compute_iou(polygons[i], polygons[ixs[1:]])
    # Identify boxes with IoU over the threshold. This
    # returns indices into ixs[1:], so add 1 to get
    # indices into ixs.
    remove_ixs = np.where(iou > threshold)[0] + 1
    # Remove indices of the picked and overlapped boxes.
    ixs = np.delete(ixs, remove_ixs)
    ixs = np.delete(ixs, 0)

  return np.array(pick, dtype=np.int32)


def compute_iou(box, boxes):
  """Calculates IoU of the given box with the array of the given boxes.
  box: a polygon
  boxes: a vector of polygons
  Note: the areas are passed in rather than calculated here for
  efficiency. Calculate once in the caller to avoid duplicate work.
  """
  # Calculate intersection areas
  iou = [box.intersection(b).area / box.union(b).area for b in boxes]

  return np.array(iou, dtype=np.float32)


def convert_format(boxes_array):
  """
  :param array: an array of shape [# bboxs, 4, 2]
  :return: a shapely.geometry.Polygon object
  """
  polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
  return np.array(polygons)


@gin.configurable
def train_eval(
    root_dir,
    experiment_name,  # experiment name
    env_name='carla-v0',
    num_iterations=int(1e7),
    actor_fc_layers=(256, 256),
    model_network_ctor_type='non-hierarchical',  # model net
    input_names=['camera', 'lidar'],  # names for inputs
    reconstruct_names=['roadmap'],  # names for masks
    pixor_names=['vh_clas', 'vh_regr', 'pixor_state'],
    extra_names=['state'],  # extra inputs
    obs_size=64,
    pixor_size=64,
    # Params for collect
    initial_collect_steps=100,
    replay_buffer_capacity=int(5e4+1),
    # Params for train
    model_batch_size=32,  # model training batch size
    sequence_length=4,  # number of timesteps to train model
    model_learning_rate=1e-4,  # learning rate for model training
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for summaries and logging
    num_images_per_summary=1,  # images for each summary
    train_checkpoint_interval=10000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    gpu_allow_growth=True,  # GPU memory growth
    gpu_memory_limit=None,  # GPU memory limit
    action_repeat=1):  # Name of single observation channel, ['camera', 'lidar', 'birdeye']
  """A simple train and eval for SLAC."""
  # Setup GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpu_allow_growth:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpu_memory_limit:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
          gpu,
          [tf.config.experimental.VirtualDeviceConfiguration(
              memory_limit=gpu_memory_limit)])

  # Get train and eval direction
  root_dir = os.path.expanduser(root_dir)
  root_dir = os.path.join(root_dir, env_name, experiment_name)

  # Get summary writers
  summary_writer = tf.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

  # Eval metrics
  eval_metrics = [
      tf_metrics.AverageReturnMetric(
        name='AverageReturnEvalPolicy', buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(
        name='AverageEpisodeLengthEvalPolicy',
        buffer_size=num_eval_episodes),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()

  # Whether to record for summary
  with tf.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    # Create Carla environment
    py_env, eval_py_env = load_carla_env(env_name='carla-v0', lidar_bin=32/obs_size, pixor_size=pixor_size,
      obs_channels=list(set(input_names+reconstruct_names+pixor_names+extra_names)), action_repeat=action_repeat)

    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    fps = int(np.round(1.0 / (py_env.dt * action_repeat)))

    # Specs
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    # Get model network
    if model_network_ctor_type == 'hierarchical':
      model_network_ctor = sequential_latent_pixor_network.PixorSLMHierarchical
    elif model_network_ctor_type == 'non-hierarchical':
      model_network_ctor = sequential_latent_pixor_network.PixorSLMNonHierarchical
    else:
      raise NotImplementedError
    model_net = model_network_ctor(
      input_names, reconstruct_names, obs_size=obs_size, pixor_size=pixor_size)

    # Build the perception agent
    actor_network = state_based_heuristic_actor_network.StateBasedHeuristicActorNetwork(
        observation_spec['state'],
        action_spec,
        desired_speed=9
        )

    tf_agent = perception_agent.PerceptionAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_network,
        model_network=model_net,
        model_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_learning_rate),
        num_images_per_summary=num_images_per_summary,
        sequence_length=sequence_length,
        gradient_clipping=gradient_clipping,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        fps=fps)
    tf_agent.initialize()

    # Get replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,  # No parallel environments
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    # Train metrics
    env_steps = tf_metrics.EnvironmentSteps()
    average_return = tf_metrics.AverageReturnMetric(
        buffer_size=num_eval_episodes,
        batch_size=tf_env.batch_size)
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        env_steps,
        average_return,
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size),
    ]

    # Get policies
    eval_policy = tf_agent.policy
    initial_collect_policy = tf_agent.collect_policy

    # Checkpointers
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'train'),
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        max_to_keep=2)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    # Collect driver
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps)

    # Optimize the performance by using tf functions
    initial_collect_driver.run = common.function(initial_collect_driver.run)

    # Collect initial replay data.
    if (global_step.numpy() == 0 and replay_buffer.num_frames() == 0):
      logging.info(
          'Collecting experience for %d steps '
          'with a model-based policy.', initial_collect_steps)
      initial_collect_driver.run()
      rb_checkpointer.save(global_step=global_step.numpy())

    compute_summaries(
      eval_metrics,
      eval_tf_env,
      eval_policy,
      train_step=global_step,
      summary_writer=summary_writer,
      num_episodes=1,
      num_episodes_to_render=1,
      model_net=model_net,
      fps=10,
      image_keys=['camera', 'lidar', 'roadmap'],
      pixor_size=pixor_size)

    # Dataset generates trajectories with shape [Bxslx...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=model_batch_size,
        num_steps=sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    # Get train model step
    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)
    train_step = common.function(train_step)

    # Start training
    for iteration in range(num_iterations):

      loss = train_step()

      # Log training information
      if global_step.numpy() % log_interval == 0:
        logging.info('global steps = %d, model loss = %f', global_step.numpy(), loss.loss)

      # Get training metrics
      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=global_step.numpy())

      # Evaluation
      if global_step.numpy() % eval_interval == 0:
        # Log evaluation metrics
        compute_summaries(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          train_step=global_step,
          summary_writer=summary_writer,
          num_episodes=num_eval_episodes,
          num_episodes_to_render=num_images_per_summary,
          model_net=model_net,
          fps=10,
          image_keys=['camera', 'lidar', 'roadmap'],
          pixor_size=pixor_size)

      # Save checkpoints
      global_step_val = global_step.numpy()
      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(FLAGS.root_dir, FLAGS.experiment_name)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  flags.mark_flag_as_required('experiment_name')
  app.run(main)
