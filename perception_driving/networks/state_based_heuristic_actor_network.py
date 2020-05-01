# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gin
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common


@gin.configurable
class StateBasedHeuristicActorNetwork(network.Network):
  """A heuristic actor network for autonomous vehicle."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               desired_speed=8,
               add_noise=False,
               name='StateBasedHeuristicActorNetwork'):
    """Creates an instance of `StateBasedHeuristicActorNetwork`.
    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        inputs.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the outputs.
      desired_speed: The desired speed of the vehicle to track.
      name: A string representing name of the network.
    """
    super(StateBasedHeuristicActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._output_tensor_spec = output_tensor_spec
    self._desired_speed = desired_speed
    self._add_noise = add_noise

    # if add_noise:
    #   self._t = 0
    #   self._noise_interval = 80
    #   self._noise_duration = 10
    #   self._noise_stddev = [1.0, 0.5]  # [acc, steer]

  def call(self, inputs, step_type=(), network_state=(), training=False):
    del step_type, training  # unused.

    inputs = tf.cast(tf.nest.flatten(inputs)[0], tf.float32)
    dis = tf.gather(inputs, 0, axis=-1)
    ang = tf.gather(inputs, 1, axis=-1)
    speed = tf.gather(inputs, 2, axis=-1)
    front = tf.cast(tf.gather(inputs, 3, axis=-1), dtype=tf.bool)

    accel = - 1.0 * (speed - self._desired_speed)
    steer =  - 0.2 * dis - 2.0 * ang

    accel_stop = - 8.0 * tf.ones_like(speed)
    accel = tf.where(front, accel_stop, accel)

    # # Add action noise
    # if self._add_noise:
    #   if self._t%self._noise_interval < self._noise_duration:
    #     accel += tf.random.normal(accel.shape, stddev=self._noise_stddev[0])
    #     steer += tf.random.normal(steer.shape, stddev=self._noise_stddev[1])
    #   self._t += 1

    actions = tf.transpose(tf.stack((accel, steer)))

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                              [actions])

    return output_actions, network_state