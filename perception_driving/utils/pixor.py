# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/philip-huang/PIXOR>:
# Copyright (c) [2019] [Yizhou Huang]
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Utils for PIXOR detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import pickle

import numpy as np
import tensorflow as tf

from shapely.geometry import Polygon


def get_eval_lists(images, latents, model_net, pixor_size=128):
  gt_boxes_list = []
  corners_list = []
  scores_list = []
  gt_pixor_state_list = []
  recons_pixor_state_list = []
  for i in range(len(latents)):
    latent_eps = latents[i]
    image_eps = images[i]
    for j in range(len(latent_eps)):
      latent = latent_eps[j]
      dict_obs = image_eps[j]
      dict_recons = model_net.reconstruct_pixor(latent)

      vh_clas_recons = tf.squeeze(dict_recons['vh_clas'], axis=-1)  # (B,H,W,1)
      vh_regr_recons = dict_recons['vh_regr']  # (B,H,W,6)
      decoded_reg_recons = decode_reg(vh_regr_recons, pixor_size)  # (B,H,W,8)      
      pixor_state_recons = dict_recons['pixor_state']

      vh_clas_obs = tf.squeeze(dict_obs['vh_clas'], axis=-1)  # (B,H,W,1)
      vh_regr_obs = dict_obs['vh_regr']  # (B,H,W,6)
      decoded_reg_obs = decode_reg(vh_regr_obs, pixor_size)  # (B,H,W,8)
      pixor_state_obs = dict_obs['pixor_state']

      B = vh_regr_obs.shape[0]
      for k in range(B):
        gt_boxes, _ = pixor_postprocess(vh_clas_obs[k], decoded_reg_obs[k])
        corners, scores = pixor_postprocess(vh_clas_recons[k], decoded_reg_recons[k])  # (N,4,2)      
        gt_boxes_list.append(gt_boxes)
        corners_list.append(corners)
        scores_list.append(scores)
        gt_pixor_state_list.append(pixor_state_obs[k])
        recons_pixor_state_list.append(pixor_state_recons[k])
  return gt_boxes_list, corners_list, scores_list, gt_pixor_state_list, recons_pixor_state_list


def get_eval_metrics(images, latents, model_net, pixor_size=128, ap_range=[0.3,0.5,0.7], filename = 'metrics'):
  gt_boxes_list, corners_list, scores_list, gt_pixor_state_list, recons_pixor_state_list \
    = get_eval_lists(images, latents, model_net, pixor_size=pixor_size)

  N = len(gt_boxes_list)  

  APs = {}
  precisions = {}
  recalls = {}
  for ap in ap_range:
    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    for i in range(N):
      gt_boxes = gt_boxes_list[i]
      corners = corners_list[i]
      scores = scores_list[i]
      gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=ap)
      num_gt = gt_boxes.shape[0]
      num_pred = len(scores)
      
      gts += num_gt
      preds += num_pred
      all_scores.extend(list(scores))
      all_matches.extend(list(pred_match))
  
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    if gts == 0 or preds == 0:
      return

    AP, precision, recall, p, r = compute_ap(all_matches, gts, preds)
    print('ap', ap)
    print('precision', p)
    print('recall', r)
    APs[ap] = AP
    precisions[ap] = precision
    recalls[ap] = recall   
  
  results = {}
  results['APs'] = APs
  results['precisions'] = precisions
  results['recalls'] = recalls  

  error_position = []
  error_heading = []
  error_velocity = []
  for i in range(N):
    gt_pixor_state = gt_pixor_state_list[i]
    recons_pixor_state = recons_pixor_state_list[i]
    x0, y0, cos0, sin0, v0 = gt_pixor_state
    x, y, cos, sin, v = recons_pixor_state
    error_position.append(np.sqrt((x-x0)**2+(y-y0)**2))
    yaw0 = np.arctan2(sin0, cos0)
    cos0 = np.cos(yaw0)
    sin0 = np.sin(yaw0)
    yaw = np.arctan2(sin, cos)
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    error_heading.append(np.arccos(np.dot([cos,sin],[cos0,sin0])))
    error_velocity.append(abs(v-v0))
  
  results['error_position'] = np.mean(error_position)
  results['error_heading'] = np.mean(error_heading)
  results['error_velocity'] = np.mean(error_velocity)
  results['std_position'] = np.std(error_position)
  results['std_heading'] = np.std(error_heading)
  results['std_velocity'] = np.std(error_velocity)

  if not os.path.exists('results'):
    os.makedirs('results')
  path = os.path.join('results', filename)

  with open(path, 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)



def compute_matches(gt_boxes,
                    pred_boxes, pred_scores,
                    iou_threshold=0.5, score_threshold=0.0):
  """Finds matches between prediction and ground truth instances.
  Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
  """

  if len(pred_scores) == 0:
    return -1 * np.ones([gt_boxes.shape[0]]), np.array([]), np.array([])

  gt_class_ids = np.ones(len(gt_boxes), dtype=int)
  pred_class_ids = np.ones(len(pred_scores), dtype=int)

  # Sort predictions by score from high to low
  indices = np.argsort(pred_scores)[::-1]
  pred_boxes = pred_boxes[indices]
  pred_class_ids = pred_class_ids[indices]
  pred_scores = pred_scores[indices]

  # Compute IoU overlaps [pred_boxes, gt_boxes]
  overlaps = compute_overlaps(pred_boxes, gt_boxes)

  # Loop through predictions and find matching ground truth boxes
  match_count = 0
  pred_match = -1 * np.ones([pred_boxes.shape[0]])
  gt_match = -1 * np.ones([gt_boxes.shape[0]])
  for i in range(len(pred_boxes)):
    # Find best matching ground truth box
    # 1. Sort matches by score
    sorted_ixs = np.argsort(overlaps[i])[::-1]
    # 2. Remove low scores
    low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
    if low_score_idx.size > 0:
      sorted_ixs = sorted_ixs[:low_score_idx[0]]
    # 3. Find the match
    for j in sorted_ixs:
      # If ground truth box is already matched, go to next one
      if gt_match[j] > 0:
        continue
      # If we reach IoU smaller than the threshold, end the loop
      iou = overlaps[i, j]
      if iou < iou_threshold:
        break
      # Do we have a match?
      if pred_class_ids[i] == gt_class_ids[j]:
        match_count += 1
        gt_match[j] = i
        pred_match[i] = j
        break

  return gt_match, pred_match, overlaps


def compute_overlaps(boxes1, boxes2):
  """Computes IoU overlaps between two sets of boxes.
  boxes1, boxes2: a np array of boxes
  For better performance, pass the largest set first and the smaller second.
  :return: a matrix of overlaps [boxes1 count, boxes2 count]
  """
  # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
  # Each cell contains the IoU value.

  boxes1 = convert_format(boxes1)
  boxes2 = convert_format(boxes2)
  overlaps = np.zeros((len(boxes1), len(boxes2)))
  for i in range(overlaps.shape[1]):
    box2 = boxes2[i]
    overlaps[:, i] = compute_iou(box2, boxes1)
  return overlaps


def compute_ap(pred_match, num_gt, num_pred):

  assert num_gt != 0
  assert num_pred != 0
  tp = (pred_match > -1).sum()
  # Compute precision and recall at each prediction box step
  precisions = np.cumsum(pred_match > -1) / (np.arange(num_pred) + 1)
  recalls = np.cumsum(pred_match > -1).astype(np.float32) / num_gt

  # Ensure precision values decrease but don't increase. This way, the
  # precision value at each recall threshold is the maximum it can be
  # for all following recall thresholds, as specified by the VOC paper.
  for i in range(len(precisions) - 2, -1, -1):
    precisions[i] = np.maximum(precisions[i], precisions[i + 1])

  # Compute mean AP over recall range
  indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
  mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
  precision = tp / num_pred
  recall = tp / num_gt
  return mAP, precisions, recalls, precision, recall


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


def pixor_postprocess(vh_clas, decoded_reg, cls_threshold=0.6):
  """Return bounding-box image with shape (H, W, 3).
    
    vh_clas: (H, W) tensor
    decoded_reg: (H, W, 8) tensor
  """
  activation = vh_clas.numpy() > cls_threshold
  num_boxes = int(activation.sum())

  if num_boxes == 0:  # No bounding boxes
    return np.array([]), []

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