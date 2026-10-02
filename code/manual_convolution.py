from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int]=[1,1,1,1], padding = "VALID", use_bias = True, trainable=True, *args, **kwargs):
        """
        :param filter_shape: list of [filter_height, filter_width, in_channels, out_channels]
        :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
        :param padding: either "SAME" or "VALID", capitalization matters
        """
        super().__init__()

        self.strides = strides
        self.padding = padding

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable = trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable)
        self.use_bias = use_bias
        if use_bias: self.bias = get_var("conv_bias", [filter_shape[-1]], trainable)
        else: self.bias = None

    def get_weights(self):
        if self.bias is not None: return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None): 
        self.filters = filters
        if bias is not None: self.bias = bias

    def call(self, inputs):
        """
        :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
        """

        #define some useful variables
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape

        # fill out the rest!
        # check stride == 1
        assert self.strides == [1, 1, 1, 1]

        # check input channels == filter input channels
        assert input_in_channels == filter_in_channels

        # padding
        if self.padding == "SAME":
            total_pad_y = filter_height - 1
            total_pad_x = filter_width - 1

            # if odd padding put less on top/left
            pad_top = total_pad_y // 2
            pad_bottom = total_pad_y - pad_top

            pad_left = total_pad_x // 2
            pad_right = total_pad_x - pad_left

        elif self.padding == "VALID":
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        else:
            raise ValueError("padding must be 'SAME' or 'VALID'")

        padded_inputs = tf.pad(
            inputs,
            [
                [0, 0],
                [pad_top, pad_bottom],
                [pad_left, pad_right],
                [0, 0]
            ]
        )

        # output dim
        # stride = 1
        output_height = (in_height + pad_top + pad_bottom - filter_height) + 1

        output_width = (in_width + pad_left + pad_right - filter_width) + 1

        # conv
        output_rows = []

        for y in range(output_height):
            output_cols = []

            for x in range(output_width):

                # shape = [batch, filter_height, filter_width, input_channels]
                patch = padded_inputs[
                    :,
                    y:y + filter_height,
                    x:x + filter_width,
                    :
                ]

                # output channel dim=[batch, fh, fw, in_channels, 1]
                patch = tf.expand_dims(patch, axis=-1)

                # filters = [fh, fw, in_channels, out_channels]
                # broadcasting shape = [batch, fh, fw, in_channels, out_channels]
                multiplied = tf.multiply(patch, self.filters)

                # sum over fh, fw, and input channels
                # shape = [batch, out_channels]
                conv_value = tf.reduce_sum(multiplied, axis=[1, 2, 3])

                # bias
                if self.use_bias:
                    conv_value = conv_value + self.bias

                output_cols.append(conv_value)

            # [batch, output_width, out_channels]
            output_row = tf.stack(output_cols, axis=1)
            output_rows.append(output_row)

        # [batch, output_height, output_width, out_channels]
        output = tf.stack(output_rows, axis=1)

        return tf.convert_to_tensor(
            output,
            dtype=tf.float32
        )