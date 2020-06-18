"""
Convolutional layer
"""

import tensorflow as tf
from tensorflow import keras

from speenai import SpeenaiError
from speenai.utils.tensor import conv_output_length


class DepthwiseSeparableConv1D(keras.layers.Conv1D):
    """
    Depthwise separable 1D convolution

    Reference
    F. Chollet, Xception: Deep learning with depthwise separable convo- lutions,
    in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 1251–1258.
    """

    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 name=None,
                 **kwargs):
        super(DepthwiseSeparableConv1D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = keras.initializers.get(
            depthwise_initializer)
        self.depthwise_regularizer = keras.regularizers.get(
            depthwise_regularizer)
        self.depthwise_constraint = keras.constraints.get(depthwise_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.strides = [1, stride, 1, 1]

        self.depthwise_kernel = None
        self.bias = None

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise SpeenaiError(
                f'Inputs to `DepthwiseSeparableConv1D` should have rank 3. '
                f'Received input shape: {input_shape}')
        channel_axis = 1 if self.data_format == 'channels_first' else 2
        if input_shape.dims[channel_axis].value is None:
            raise SpeenaiError('The channel dimension of the inputs to '
                               '`DepthwiseSeparableConv1D` should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], 1, input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        # call at the end
        self.built = True

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -2)
        data_format = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
        outputs = tf.nn.depthwise_conv2d(inputs,
                                         self.depthwise_kernel,
                                         strides=self.strides,
                                         padding=self.padding.upper(),
                                         rate=self.dilation_rate,
                                         data_format=data_format)
        if self.use_bias:
            outputs = keras.backend.bias_add(outputs,
                                             self.bias,
                                             data_format=self.data_format)
        outputs = tf.squeeze(outputs, [-2])
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        out_filters = input_shape[-1] * self.depth_multiplier
        length = conv_output_length(input_shape[1], self.kernel_size,
                                    self.padding, self.strides)
        return input_shape[0], length, out_filters

    @classmethod
    def get_input_shape(cls, input_shape):
        input_shape = [input_shape[0], input_shape[1], 1, input_shape[2]]
        return input_shape
