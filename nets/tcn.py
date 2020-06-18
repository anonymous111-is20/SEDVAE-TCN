import tensorflow as tf
from tensorflow import keras

from speenai.models.layers import layer_norm, DepthwiseSeparableConv1D


class TemporalConvNet(keras.Model):
    """
    Temporal convolutional network for estimating separation mask

    Reference: Luo, Yi, and Nima Mesgarani. "Conv-tasnet: Surpassing ideal time-frequency magnitude
    masking for speech separation." TASLP 27.8 (2019): 1256-1266. Figure 1B
    """
    def __init__(self, encoder_n_filters, bottleneck_n_channels, convolution_n_channels,
                 convolution_kernel_size, n_blocks, n_repeats,
                 n_sources, causal, norm_type, block_type='tcn', n_inv_repeats=0, name=None):
        """
        Initialization
        :param encoder_n_filters:
        :param bottleneck_n_channels:
        :param convolution_n_channels:
        :param convolution_kernel_size:
        :param n_blocks:
        :param n_repeats:
        :param n_sources
        :param causal
        :param norm_type
        :param block_type:
        :param n_inv_repeats:
        :param name:
        """
        super(TemporalConvNet, self).__init__(name=name)
        self._encoder_n_filters = encoder_n_filters
        self._bottleneck_n_channels = bottleneck_n_channels
        self._convolution_n_channels = convolution_n_channels
        self._convolution_kernel_size = convolution_kernel_size
        self._block_type = block_type
        self._n_blocks = n_blocks
        self._n_inv_repeats = n_inv_repeats
        self._n_repeats = n_repeats
        self._n_sources = n_sources
        self._causal = causal
        self._norm_type = norm_type
        # layer norm
        self.layer_norm = layer_norm(norm_type=norm_type,
                                     name='layer_norm_input')
        # 1x1-conv bottleneck layer
        self.conv_bottleneck = keras.layers.Conv1D(filters=bottleneck_n_channels,
                                                   kernel_size=1,
                                                   name='bottleneck_layer')
        # dilated convolution layer
        self.dilated_conv_blocks = self._dilated_convolutional_blocks()

        self.conv_out = keras.layers.Conv1D(filters=encoder_n_filters * self._n_sources,
                                            kernel_size=1,
                                            name='conv1d_out')
        self.norm_out = layer_norm(norm_type=norm_type, name='norm_out')
        self.prelu = keras.layers.PReLU(shared_axes=[1], name='prelu_out')

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: of shape [batch_size, length, encoder_n_filters]
        :param training:
        :param mask:
        :return:
        """
        input_shape = tf.shape(inputs)
        # first normalize the input
        x = self.layer_norm(inputs)
        # 1x1-D convolution bottleneck layer. This block determines the number of channels
        # in the input and residual path of the subsequent convolutional blocks.
        x = self.conv_bottleneck(x)
        # dilated temporal convolutional blocks
        skip_connections = 0.
        for dilated_conv_block in self.dilated_conv_blocks:
            # x, skip = dilated_conv_block(x, training=training)
            residual, skip = dilated_conv_block(x, training=training)
            x = x + residual
            skip_connections = skip_connections + skip
        x = self.prelu(skip_connections)
        x = self.conv_out(x)
        masks = tf.reshape(x, [input_shape[0],
                               input_shape[1],
                               self._n_sources,
                               self._encoder_n_filters])
        masks = tf.nn.softmax(masks, axis=2)
        return masks

    def _dilated_convolutional_blocks(self):
        blocks = []
        if self._block_type == 'inv-some-blocks':
            for r in range(self._n_inv_repeats):
                for n in range(self._n_blocks):
                    dilation = 2 * (self._n_blocks - n - 1)
                    blocks.append(TemporalConvBlock(channels=self._bottleneck_n_channels,
                                                    filters=self._convolution_n_channels,
                                                    kernel_size=self._convolution_kernel_size,
                                                    dilation=dilation,
                                                    causal=self._causal,
                                                    norm_type=self._norm_type,
                                                    name=f'mini_block_{r}/conv_block_{n}'))
            for r in range(self._n_inv_repeats, self._n_repeats):
                for n in range(self._n_blocks):
                    dilation = 2 ** n
                    blocks.append(TemporalConvBlock(channels=self._bottleneck_n_channels,
                                                    filters=self._convolution_n_channels,
                                                    kernel_size=self._convolution_kernel_size,
                                                    dilation=dilation,
                                                    causal=self._causal,
                                                    norm_type=self._norm_type,
                                                    name=f'mini_block_{r}/conv_block_{n}'))
        else:
            for r in range(self._n_repeats):
                for n in range(self._n_blocks):
                    dilation = 2 ** n if self._block_type == 'tcn' else 2 * (self._n_blocks - n - 1)
                    blocks.append(TemporalConvBlock(channels=self._bottleneck_n_channels,
                                                    filters=self._convolution_n_channels,
                                                    kernel_size=self._convolution_kernel_size,
                                                    dilation=dilation,
                                                    causal=self._causal,
                                                    norm_type=self._norm_type,
                                                    name=f'mini_block_{r}/conv_block_{n}'))
        return blocks


class TemporalConvBlock(tf.keras.Model):
    """
    Temporal convolutional block

    Reference: Luo, Yi, and Nima Mesgarani. "Conv-tasnet: Surpassing ideal time-frequency magnitude
    masking for speech separation." TASLP 27.8 (2019): 1256-1266.

    Figure 1C
    """
    def __init__(self, channels, filters, kernel_size,
                 dilation, causal, norm_type, name=None):
        """
        :param channels:
        :param filters:
        :param kernel_size:
        :param dilation:
        :param causal:
        :param norm_type:
        :param name:
        """
        super(TemporalConvBlock, self).__init__(name=name)
        self._channels = channels
        self._filters = filters
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._causal = causal

        # first block
        self.first_1x1_conv = keras.layers.Conv1D(filters=filters,
                                                  kernel_size=1,
                                                  name='first_1x1_conv')
        self.first_prelu = keras.layers.PReLU(shared_axes=[1], name='first_prelu')
        self.first_norm = layer_norm(norm_type=norm_type, name='first_norm')

        # depthwise layer
        self.depthwise_conv = DepthwiseSeparableConv1D(kernel_size=kernel_size,
                                                       dilation_rate=dilation,
                                                       name='depthwise_conv1d')
        # sencond block
        self.second_prelu = keras.layers.PReLU(shared_axes=[1], name='second_prelu')
        self.second_norm = layer_norm(norm_type=norm_type, name='second_norm')

        # two outputs
        self.residual_conv = keras.layers.Conv1D(filters=channels,
                                                 kernel_size=1,
                                                 name='residual_conv1d')
        self.skip_conv = keras.layers.Conv1D(filters=channels,
                                             kernel_size=1,
                                             name='skip_conv1d')

    def call(self, inputs, training=None, mask=None):
        # first block
        x = self.first_1x1_conv(inputs)
        x = self.first_prelu(x)
        x = self.first_norm(x, training=training)

        # padding
        padding_size = self._dilation * (self._kernel_size - 1)
        if self._causal:
            padding = [[0, 0], [padding_size, 0], [0, 0]]
        else:
            padding = [[0, 0], [padding_size // 2, padding_size // 2], [0, 0]]
        x = tf.pad(x, padding)

        # depthwise convolutional block
        x = self.depthwise_conv(x)

        # 2nd block
        x = self.second_prelu(x)
        x = self.second_norm(x, training=training)

        # 2 paths outputs
        residual = self.residual_conv(x)
        skip = self.skip_conv(x)

        return residual, skip

    def compute_output_shape(self, input_shape):
        # the output has the same size as input
        return input_shape, input_shape
