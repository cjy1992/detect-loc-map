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

  def __init__(self, base_depth, reconstruct_pixor_state=True, name=None):
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

    if reconstruct_pixor_state:
      self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
      self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
      self.state_output_layer = tf.keras.layers.Dense(5)

    self.reconstruct_pixor_state = reconstruct_pixor_state

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

    pixor = (vh_clas, vh_regr)

    if self.reconstruct_pixor_state:
      pixor_state = self.dense1(latent)
      pixor_state = self.dense2(pixor_state)
      pixor_state = self.state_output_layer(pixor_state)  # (..., 5)
      pixor = (vh_clas, vh_regr, pixor_state)

    return pixor


class PixorDecoder128(tf.Module):
  """Decoder from latent to PIXOR outputs."""

  def __init__(self, base_depth, reconstruct_pixor_state=True, name=None):
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

    if reconstruct_pixor_state:
      self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
      self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
      self.state_output_layer = tf.keras.layers.Dense(5)

    self.reconstruct_pixor_state = reconstruct_pixor_state

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

    pixor = (vh_clas, vh_regr)

    if self.reconstruct_pixor_state:
      pixor_state = self.dense1(latent)
      pixor_state = self.dense2(pixor_state)
      pixor_state = self.state_output_layer(pixor_state)  # (..., 5)
      pixor = (vh_clas, vh_regr, pixor_state)

    return pixor


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
  """The sequential latent model that also reconstruct PIXOR outputs."""

  def __init__(self,
               input_names,
               reconstruct_names,
               obs_size=64,
               pixor_size=64,
               reconstruct_pixor_state=True,
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               kl_analytic=True,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               name=None):
    """Creates an instance of `PixorSLMHierarchical`.
    Args:
      input_names: the names of the observation inputs (e.g, 'camera', 'lidar').
      reconstruct_names: names of the outputs to reconstruct (e.g, 'mask').
      obs_size: the pixel size of the observation inputs. Here we assume
        the image inputs have same width and height.
      pixor_size: the pixel size of the PIXOR outputs. Here we assume
        the images have same width and height.
      base_depth: base depth of the convolutional layers.
      latent1_size: size of the first latent of the hierarchical latent model.
      latent2_size: size of the second latent of the hierarchical latent model.
      kl_analytic: whether to use analytical KL divergence.
      decoder_stddev: standard deviation of the decoder.
      name: A string representing name of the network.
    """
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
    self.reconstruct_pixor_state = reconstruct_pixor_state

    if pixor_size == 64:
      self.pixor_decoder = PixorDecoder64(base_depth, reconstruct_pixor_state)
    elif pixor_size == 128:
      self.pixor_decoder = PixorDecoder128(base_depth, reconstruct_pixor_state)
    else:
      raise NotImplementedError

  def get_features(self, images):
    features = {}
    for name in self.input_names:
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      features[name] = self.encoders[name](images_tmp)
    features = tf.concat(list(features.values()), axis=-1)
    return features

  def reconstruct_pixor(self, latent):
    if self.reconstruct_pixor_state:
      vh_clas, vh_regr, pixor_state = self.pixor_decoder(latent)
      return {'vh_clas': vh_clas, 'vh_regr': vh_regr, 'pixor_state': pixor_state}
    else:
      vh_clas, vh_regr = self.pixor_decoder(latent)
      return {'vh_clas': vh_clas, 'vh_regr': vh_regr, 'pixor_state': tf.zeros((vh_clas.shape[0], 5))}

  def compute_loss(self, images, actions, step_types, latent_posterior_samples_and_dists=None, num_first_image=5):
    # Compuate the latents
    latent1_dists, latent2_dists, latent1_samples, latent2_samples = \
      self.compute_latents(images, actions, step_types, latent_posterior_samples_and_dists, num_first_image)

    latent1_posterior_dists, latent1_prior_dists = latent1_dists
    latent2_posterior_dists, latent2_prior_dists = latent2_dists
    latent1_posterior_samples, latent1_prior_samples, \
      latent1_conditional_prior_samples = latent1_samples
    latent2_posterior_samples, latent2_prior_samples, \
      latent2_conditional_prior_samples = latent2_samples

    # Compute the KL divergence part of the ELBO
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

    elbo = - latent1_kl_divergences - latent2_kl_divergences

    # Compute the reconstruction part of the ELBO
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

    # Compute the loss of KL divergence and reconstruction
    loss = -tf.reduce_mean(elbo)

    # Generate the images
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

    # Compute the perception loss
    if self.reconstruct_pixor_state:
      vh_clas_pred, vh_regr_pred, pixor_state_pred = self.pixor_decoder(latent1_posterior_samples, latent2_posterior_samples)
      pixor_state_label = tf.image.convert_image_dtype(images['pixor_state'], tf.float32)
    else:
      vh_clas_pred, vh_regr_pred = self.pixor_decoder(latent1_posterior_samples, latent2_posterior_samples)
    vh_clas_label = tf.image.convert_image_dtype(images['vh_clas'], tf.float32)
    vh_regr_label = tf.image.convert_image_dtype(images['vh_regr'], tf.float32)

    cls_loss = focal_loss(vh_clas_pred, vh_clas_label)
    reg_loss = smoothL1_loss(vh_clas_label*vh_regr_pred, vh_regr_label) * 0.1
    outputs.update({
      'cls_loss': cls_loss,
      'reg_loss': reg_loss,
    })

    perception_loss = cls_loss + reg_loss

    if self.reconstruct_pixor_state:
      state_loss = smoothL1_loss(pixor_state_pred, pixor_state_label)
      perception_loss += state_loss
      outputs.update({'state_loss': state_loss})

    loss = loss + 1 * perception_loss

    return loss, outputs