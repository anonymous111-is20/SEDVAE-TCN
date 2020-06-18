"""
Normalization layer
"""
import tensorflow as tf
from tensorflow import keras

from speenai import EPS, SpeenaiError


class ChannelLayerNorm(keras.layers.Layer):
    """
    Cummulative Layer Normalization
    """
    def __init__(self, **kwargs):
        super(ChannelLayerNorm, self).__init__(**kwargs)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(name='beta',
                                    shape=tf.TensorShape(input_shape[-1]),
                                    initializer=tf.zeros_initializer,
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=tf.TensorShape(input_shape[-1]),
                                     initializer=tf.ones_initializer,
                                     trainable=True)
        # Be sure to call this at the end
        super(ChannelLayerNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        means, variances = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        x = self.gamma * (inputs - means) / tf.pow(variances + EPS,
                                                   0.5) + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        base_config = super(ChannelLayerNorm, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GlobalLayerNorm(keras.layers.Layer):
    """
    Global layer normalization
    """

    def __init__(self, **kwargs):
        super(GlobalLayerNorm, self).__init__(**kwargs)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(name='beta',
                                    shape=tf.TensorShape(input_shape[-1]),
                                    initializer=tf.zeros_initializer,
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=tf.TensorShape(input_shape[-1]),
                                     initializer=tf.ones_initializer,
                                     trainable=True)
        # Be sure to call this at the end
        super(GlobalLayerNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        means, variances = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        x = self.gamma * (inputs - means) / tf.pow(variances + EPS,
                                                   0.5) + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        base_config = super(GlobalLayerNorm, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def layer_norm(norm_type, name=None):
    """
    Layer normalization
    :param norm_type: normalization type
    :param name: layer name
    :return:
    """
    if norm_type == 'gln':
        return GlobalLayerNorm(name=name)
    elif norm_type == 'cln':
        return ChannelLayerNorm(name=name)
    else:
        raise SpeenaiError(f'Unknown normalization type {norm_type}')
