# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

from interp_e2e_driving.networks import sequential_latent_network
from interp_e2e_driving.utils import nest_utils

tfd = tfp.distributions
EPS = 1e-8


class PixorDecoder64(tf.Module):
  """Decoder from latent to PIXOR outputs."""

  def __init__(self, base_depth, name=None):
    super(PixorDecoder64, self).__init__(name=name)
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
    # self.conv_transpose5 = conv_transpose(channels, 5, 2)
    self.conv_transpose_cls = tf.keras.layers.Conv2DTranspose(
        1, 5, 2, padding="SAME", activation=tf.nn.sigmoid)
    self.conv_transpose_reg = conv_transpose(6, 5, 2)

    self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
    self.state_output_layer = tf.keras.layers.Dense(5)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    # out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)
    vh_clas = self.conv_transpose_cls(out)  # (sample*N*T, h, w, 1)
    vh_regr = self.conv_transpose_reg(out)  # (sample*N*T, h, w, 6)

    expanded_shape_cls = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(vh_clas)[1:]], axis=0)
    vh_clas = tf.reshape(vh_clas, expanded_shape_cls)  # (sample, N, T, h, w, 1)
    
    expanded_shape_reg = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(vh_regr)[1:]], axis=0)
    vh_regr = tf.reshape(vh_regr, expanded_shape_reg)  # (sample, N, T, h, w, 6)

    pixor_state = self.dense1(latent)
    pixor_state = self.dense2(pixor_state)
    pixor_state = self.state_output_layer(pixor_state)  # (..., 5)

    return vh_clas, vh_regr, pixor_state


class PixorDecoder128(tf.Module):
  """Decoder from latent to PIXOR outputs."""

  def __init__(self, base_depth, name=None):
    super(PixorDecoder128, self).__init__(name=name)
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(8 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(base_depth, 3, 2)
    # self.conv_transpose5 = conv_transpose(channels, 5, 2)
    self.conv_transpose_cls = tf.keras.layers.Conv2DTranspose(
        1, 5, 2, padding="SAME", activation=tf.nn.sigmoid)
    self.conv_transpose_reg = conv_transpose(6, 5, 2)

    self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
    self.state_output_layer = tf.keras.layers.Dense(5)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)
    # out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)
    vh_clas = self.conv_transpose_cls(out)  # (sample*N*T, h, w, 1)
    vh_regr = self.conv_transpose_reg(out)  # (sample*N*T, h, w, 6)

    expanded_shape_cls = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(vh_clas)[1:]], axis=0)
    vh_clas = tf.reshape(vh_clas, expanded_shape_cls)  # (sample, N, T, h, w, 1)
    
    expanded_shape_reg = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(vh_regr)[1:]], axis=0)
    vh_regr = tf.reshape(vh_regr, expanded_shape_reg)  # (sample, N, T, h, w, 6)

    pixor_state = self.dense1(latent)
    pixor_state = self.dense2(pixor_state)
    pixor_state = self.state_output_layer(pixor_state)  # (..., 5)

    return vh_clas, vh_regr, pixor_state


def focal_loss(pred, label, alpha=0.25, gamma=2.0):
  loss = - alpha * (1-pred)**gamma * tf.math.log(pred + EPS) * label
  loss += - (1 - alpha) * pred**gamma * tf.math.log(1 - pred + EPS) * (1 - label)
  loss = tf.reduce_sum(loss)
  return loss


def smoothL1_loss(pred, label):
  abs_diff = abs(pred - label)
  below_one = tf.cast(abs_diff < 1, dtype=tf.float32)
  above_one = 1 - below_one
  loss = 0.5 * abs_diff**2 * below_one + (abs_diff - 0.5) * above_one
  loss = tf.reduce_sum(loss)
  return loss


@gin.configurable
class PixorSLMHierarchical(
  sequential_latent_network.SequentialLatentModelHierarchical):

  def __init__(self,
               input_names,
               reconstruct_names,
               obs_size=64,
               pixor_size=64,
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               kl_analytic=True,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               name=None):
    super(PixorSLMHierarchical, self).__init__(
      input_names=input_names,
      reconstruct_names=reconstruct_names,
      obs_size=obs_size,
      base_depth=base_depth,
      latent1_size=latent1_size,
      latent2_size=latent2_size,
      kl_analytic=kl_analytic,
      decoder_stddev=decoder_stddev,
      name=name)
    self.pixor_size = pixor_size

    if pixor_size == 64:
      self.pixor_decoder = PixorDecoder64(base_depth)
    elif pixor_size == 128:
      self.pixor_decoder = PixorDecoder128(base_depth)
    else:
      raise NotImplementedError

  def get_features(self, images):
    features = {}
    for name in self.input_names:
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      features[name] = self.encoders[name](images_tmp)
    # features = sum(features.values())
    features = tf.concat(list(features.values()), axis=-1)
    return features

  def reconstruct_pixor(self, latent):
    vh_clas, vh_regr, pixor_state = self.pixor_decoder(latent)
    return {'vh_clas': vh_clas, 'vh_regr': vh_regr, 'pixor_state': pixor_state}

  def compute_loss(self, images, actions, step_types, latent_posterior_samples_and_dists=None):
    sequence_length = step_types.shape[1] - 1

    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
    (latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = (
        latent_posterior_samples_and_dists)
    (latent1_prior_samples, latent2_prior_samples), _ = self.sample_prior_or_posterior(actions, step_types)  # for visualization

    first_image = {}
    num_first_image = 5
    for k,v in images.items():
      first_image[k] = v[:, :num_first_image]
    (latent1_conditional_prior_samples, latent2_conditional_prior_samples), _ = self.sample_prior_or_posterior(
        actions, step_types, images=first_image)  # for visualization. condition on first image only

    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                             tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)

    latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent1_size])
    latent1_first_prior_dists = self.latent1_first_prior(step_types)
    # these distributions start at t=1 and the inputs are from t-1
    latent1_after_first_prior_dists = self.latent1_prior(
        latent2_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent1_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent1_reset_masks),
        latent1_first_prior_dists,
        latent1_after_first_prior_dists)

    latent2_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent2_size])
    latent2_first_prior_dists = self.latent2_first_prior(latent1_posterior_samples)
    # these distributions start at t=1 and the last 2 inputs are from t-1
    latent2_after_first_prior_dists = self.latent2_prior(
        latent1_posterior_samples[:, 1:sequence_length+1],
        latent2_posterior_samples[:, :sequence_length],
        actions[:, :sequence_length])
    latent2_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent2_reset_masks),
        latent2_first_prior_dists,
        latent2_after_first_prior_dists)

    outputs = {}

    if self.kl_analytic:
      latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
    else:
      latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples)
                                - latent1_prior_dists.log_prob(latent1_posterior_samples))
    latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)
    outputs.update({
      'latent1_kl_divergence': tf.reduce_mean(latent1_kl_divergences),
    })

    if self.kl_analytic:
      latent2_kl_divergences = tfd.kl_divergence(latent2_posterior_dists, latent2_prior_dists)
    else:
      latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples)
                                - latent2_prior_dists.log_prob(latent2_posterior_samples))
    latent2_kl_divergences = tf.reduce_sum(latent2_kl_divergences, axis=1)
    outputs.update({
      'latent2_kl_divergence': tf.reduce_mean(latent2_kl_divergences),
    })

    outputs.update({
      'kl_divergence': tf.reduce_mean(latent1_kl_divergences + latent2_kl_divergences),
    })

    # Compute the loss
    elbo = - latent1_kl_divergences - latent2_kl_divergences

    likelihood_dists = {}
    likelihood_log_probs = {}
    reconstruction_error = {}

    for name in self.reconstruct_names:
      likelihood_dists[name] = self.decoders[name](latent1_posterior_samples, latent2_posterior_samples)
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      likelihood_log_probs[name] = likelihood_dists[name].log_prob(images_tmp)
      likelihood_log_probs[name] = tf.reduce_sum(likelihood_log_probs[name], axis=1)
      reconstruction_error[name] = tf.reduce_sum(tf.square(images_tmp - likelihood_dists[name].distribution.loc),
                                         axis=list(range(-len(likelihood_dists[name].event_shape), 0)))
      reconstruction_error[name] = tf.reduce_sum(reconstruction_error[name], axis=1)
      outputs.update({
        'log_likelihood_'+name: tf.reduce_mean(likelihood_log_probs[name]),
        'reconstruction_error_'+name: tf.reduce_mean(reconstruction_error[name]),
      })
      elbo += likelihood_log_probs[name]

    # average over the batch dimension
    loss = -tf.reduce_mean(elbo)

    # Save the images for inputs and masks
    posterior_images = {}
    prior_images = {}
    conditional_prior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = likelihood_dists[name].mean()
      prior_images[name] = self.decoders[name](latent1_prior_samples, latent2_prior_samples).mean()
      conditional_prior_images[name] = self.decoders[name](latent1_conditional_prior_samples, latent2_conditional_prior_samples).mean()

    original_images = tf.concat([tf.image.convert_image_dtype(images[k], tf.float32)
      for k in list(set(self.input_names+self.reconstruct_names))], axis=-2)
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    prior_images = tf.concat(list(prior_images.values()), axis=-2)
    conditional_prior_images = tf.concat(list(conditional_prior_images.values()), axis=-2)

    outputs.update({
      'elbo': tf.reduce_mean(elbo),
      'original_images': original_images,
      'posterior_images': posterior_images,
      'prior_images': prior_images,
      'conditional_prior_images': conditional_prior_images,
    })

    # Compute the pixor loss
    vh_clas_pred, vh_regr_pred, pixor_state_pred = self.pixor_decoder(latent1_posterior_samples, latent2_posterior_samples)
    vh_clas_label = tf.image.convert_image_dtype(images['vh_clas'], tf.float32)
    vh_regr_label = tf.image.convert_image_dtype(images['vh_regr'], tf.float32)
    pixor_state_label = tf.image.convert_image_dtype(images['pixor_state'], tf.float32)

    # pos_pixels = tf.reduce_sum(vh_clas_label)
    cls_loss = focal_loss(vh_clas_pred, vh_clas_label)
    reg_loss = smoothL1_loss(vh_clas_label*vh_regr_pred, vh_regr_label) * 0.1
    state_loss = smoothL1_loss(pixor_state_pred, pixor_state_label)

    perception_loss = cls_loss + reg_loss + state_loss
    loss = loss + 10 * perception_loss

    # Generate cls images
    if self.obs_size == self.pixor_size:
      tile_shape = [1]*len(vh_clas_label.shape)
      tile_shape[-1] = 3
      original_images_clas = tf.tile(vh_clas_label, tile_shape)
      posterior_images_clas = tf.tile(vh_clas_pred, tile_shape)
      original_images = tf.concat([original_images, original_images_clas], axis=-2)
      posterior_images = tf.concat([posterior_images, posterior_images_clas], axis=-2)

    outputs.update({
      'cls_loss': cls_loss,
      'reg_loss': reg_loss,
      'state_loss': state_loss,
      'original_images': original_images,
      'posterior_images': posterior_images,
    })

    return loss, outputs