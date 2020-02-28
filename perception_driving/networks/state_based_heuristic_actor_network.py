import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common


@gin.configurable
class StateBasedHeuristicActorNetwork(network.Network):
  """Creates an actor network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               desired_speed = 8,
               name='StateBasedHeuristicActorNetwork'):
    """Creates an instance of `MlpNetwork`.
    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        inputs.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the outputs.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent', if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.
    Raises:
      ValueError: If `input_tensor_spec` or `action_spec` contains more than one
        item, or if the action data type is not `float`.
    """

    super(StateBasedHeuristicActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._output_tensor_spec = output_tensor_spec
    self._desired_speed = desired_speed

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

    actions = tf.transpose(tf.stack((accel, steer)))

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                              [actions])

    return output_actions, network_state