# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import eager_utils

from perception_driving.policies import perception_state_policy
from sequential_latent_model.utils import gif_utils
from sequential_latent_model.utils import nest_utils


@gin.configurable
class PerceptionAgent(tf_agent.TFAgent):
  """Agent with perception system."""
  
  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               model_network,
               model_optimizer,
               num_images_per_summary=1,
               sequence_length=2,
               gradient_clipping=None,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               fps=10,
               name=None):
    tf.Module.__init__(self, name=name)

    self._model_network = model_network
    self._model_optimizer = model_optimizer
    self._num_images_per_summary = num_images_per_summary
    self._gradient_clipping = gradient_clipping
    self._summarize_grads_and_vars = summarize_grads_and_vars
    self._train_step_counter = train_step_counter
    self._fps = fps

    # Build the policy
    policy = perception_state_policy.PerceptionStatePolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network,
      model_network=model_network,
      collect=False)

    super(PerceptionAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=sequence_length + 1,
        train_step_counter=train_step_counter,
        )

  def _train(self, experience, weights=None):
    # Train to minimize model loss
    with tf.GradientTape() as tape:
      images = experience.observation
      model_loss = self.model_loss(
          images,
          experience.action,
          experience.step_type,
          weights=weights)
    tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
    model_variables = list(self._model_network.variables)
    assert model_variables, 'No model variables to optimize.'
    model_grads = tape.gradient(model_loss, model_variables)
    self._apply_gradients(model_grads, model_variables, self._model_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='model_loss', data=model_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)

    total_loss = model_loss

    return tf_agent.LossInfo(loss=total_loss, extra=())

  def _apply_gradients(self, gradients, variables, optimizer):
    grads_and_vars = list(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

  def model_loss(self,
                 images,
                 actions,
                 step_types,
                 latent_posterior_samples_and_dists=None,
                 weights=None):
    with tf.name_scope('model_loss'):
      model_loss, outputs = self._model_network.compute_loss(
          images, actions, step_types,
          latent_posterior_samples_and_dists=latent_posterior_samples_and_dists)
      for name, output in outputs.items():
        if output.shape.ndims == 0:
          tf.summary.scalar(name, output, step=self.train_step_counter)
        elif output.shape.ndims == 5:
          output = output[:self._num_images_per_summary]
          output = tf.transpose(output, [1,0,2,3,4])
          output = tf.reshape(output, [output.shape[0], output.shape[1]*output.shape[2], output.shape[3], output.shape[4]])
          output = tf.expand_dims(output, axis=0)
          gif_utils.gif_summary(name, output, self._fps,
                       saturate=True, step=self.train_step_counter)
        else:
          raise NotImplementedError

      if weights is not None:
        model_loss *= weights

      model_loss = tf.reduce_mean(input_tensor=model_loss)

      return model_loss