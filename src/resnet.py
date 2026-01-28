import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from tensorflow.keras.layers import Layer, Conv2D

L2_REG = tf.keras.regularizers.L2(5e-5)
K_INIT = "glorot_uniform"

@register_keras_serializable()
class SpatialGroupedConv2D(Layer):
    """
    Spatially grouped Conv2D that splits the input spatially into overlapping tiles,
    applies a shared Conv2D to each tile, and stitches them back together.
    Produces identical output to a single full-image Conv2D (with padding='same').

    Args:
        filters (int): Number of output filters for each convolution.
        kernel_size (int or tuple): Size of the convolution kernel.
        groups_h (int): Number of groups to split along the height axis.
        groups_w (int): Number of groups to split along the width axis.
        **kwargs: Additional keyword arguments for Conv2D (e.g., activation, kernel_regularizer).
    """

    def __init__(self, filters, kernel_size, groups_h=1, groups_w=1, **kwargs):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups_h = groups_h
        self.groups_w = groups_w
        self.conv_kwargs = kwargs  # for Conv2D

    def build(self, input_shape):
        # Single shared Conv2D layer across all spatial tiles
        self.conv = Conv2D(
            self.filters,
            self.kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            **self.conv_kwargs,
        )

    def call(self, x):
        k_h = k_w = self.kernel_size
        pad_h, pad_w = k_h // 2, k_w // 2  # overlap size for SAME padding

        H, W = tf.shape(x)[1], tf.shape(x)[2]
        tile_h = tf.cast(tf.math.ceil(tf.cast(H, tf.float32) / self.groups_h), tf.int32)
        tile_w = tf.cast(tf.math.ceil(tf.cast(W, tf.float32) / self.groups_w), tf.int32)

        outputs = []
        for i in range(self.groups_h):
            row_outputs = []
            for j in range(self.groups_w):
                # Compute tile boundaries with overlap
                y0 = tf.maximum(0, i * tile_h - pad_h)
                y1 = tf.minimum(H, (i + 1) * tile_h + pad_h)
                x0 = tf.maximum(0, j * tile_w - pad_w)
                x1 = tf.minimum(W, (j + 1) * tile_w + pad_w)

                tile = x[:, y0:y1, x0:x1, :]
                tile_conv = self.conv(tile)

                # Crop overlap so tiles fit perfectly when concatenated
                crop_top = pad_h if i > 0 else 0
                crop_bottom = pad_h if i < self.groups_h - 1 else 0
                crop_left = pad_w if j > 0 else 0
                crop_right = pad_w if j < self.groups_w - 1 else 0

                if any([crop_top, crop_bottom, crop_left, crop_right]):
                    tile_conv = tile_conv[
                        :,
                        crop_top or 0 : tf.shape(tile_conv)[1] - (crop_bottom or 0),
                        crop_left or 0 : tf.shape(tile_conv)[2] - (crop_right or 0),
                        :
                    ]

                row_outputs.append(tile_conv)
            outputs.append(tf.concat(row_outputs, axis=2))
        return tf.concat(outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "groups_h": self.groups_h,
            "groups_w": self.groups_w,
        })
        return config

@register_keras_serializable()
class ChannelPoolAvg(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Compute mean across channel axis (axis=3), keep dims for broadcasting
        return tf.keras.backend.mean(inputs, axis=3, keepdims=True)

    def compute_output_shape(self, input_shape):
        # Output shape same as input except channels become 1
        return input_shape[:-1] + (1,)


@register_keras_serializable()
class ChannelPoolMax(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Compute max across channel axis (axis=3), keep dims for broadcasting
        return tf.keras.backend.max(inputs, axis=3, keepdims=True)

    def compute_output_shape(self, input_shape):
        # Output shape same as input except channels become 1
        return input_shape[:-1] + (1,)

@register_keras_serializable()
class RecomputeSubModel(tf.keras.layers.Layer):
    """  
    Wrap a Keras submodel so its forward is recomputed during backprop.
    """
    def __init__(self, submodel: tf.keras.Model, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.submodel = submodel

        def _forward_pass(x):
            return self.submodel(x)

        self.recompute_fn = tf.recompute_grad(_forward_pass)

    @tf.function(jit_compile=False)
    def call(self, inputs):
        return self.recompute_fn(inputs)

    def compute_output_shape(self, input_shape):
        return self.submodel.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "submodel": serialize_keras_object(self.submodel),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub = deserialize_keras_object(config.pop("submodel"))
        obj = cls(submodel=sub, **config)
        return obj

@register_keras_serializable()
class TimeCondLayer(Layer):
    def __init__(self, time_mask, use_crps=False, use_noise=False, **kwargs):
        """
        Args:
            time_mask: Indices of time-related features.
            use_crps: Whether CRPS-related logic should be used.
            use_noise: Whether to include noise vector (only if use_crps is True).
        """
        super().__init__(**kwargs)
        self.time_mask = time_mask
        self.use_crps = use_crps
        self.use_noise = use_noise

    def call(self, inputs):
        time_feats = tf.gather(inputs, self.time_mask, axis=-1)  # (B, H, W, 2)
        d = time_feats[:, 0, 0, :]  # (B, 2)

        if not self.use_crps:
            return d  # Case A: full d vector (lead time + ens_id)

        lead_time = d[:, -1:]  # (B, 1)
        if not self.use_noise:
            return lead_time  # Case B: only lead_time

        # Case C: CRPS + noise
        ens_id = tf.cast(tf.floor(tf.cast(d[:, 0], tf.float32) * (2**31 - 1)), tf.int32)  # (B,)
        seed = tf.stack([ens_id, ens_id ^ 0x9E3779B9], axis=1)  # (B,2)
        z = tf.random.stateless_normal([tf.shape(d)[0], 32], seed=seed, dtype=lead_time.dtype)  # (B,32)
        return tf.concat([z, lead_time], axis=1)  # (B, 33)

    def compute_output_shape(self, input_shape):
        if not self.use_crps:
            return (input_shape[0], len(self.time_mask))     # Case A
        elif not self.use_noise:
            return (input_shape[0], 1)     # Case B
        else:
            return (input_shape[0], 33)    # Case C

    def get_config(self):
        config = super().get_config()
        config.update({
            'time_mask': self.time_mask,
            'use_crps': self.use_crps,
            'use_noise': self.use_noise,
        })
        return config


@register_keras_serializable()
class ReflectPadLayer(Layer):
    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, self.padding, mode="REFLECT")

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = shape[1] + self.padding[0][0] + self.padding[0][1]
        shape[2] = shape[2] + self.padding[1][0] + self.padding[1][1]
        return tuple(shape)


@register_keras_serializable()
class OutputMaskLayer(Layer):
    def __init__(self, output_tensor_mask, **kwargs):
        super().__init__(**kwargs)
        self.output_tensor_mask = output_tensor_mask

    def call(self, inputs):
        return tf.gather(inputs, indices=self.output_tensor_mask, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (len(self.output_tensor_mask),)


@register_keras_serializable()
class ChannelSliceLayer(Layer):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[
            :, :, :, self.start : self.end,
        ]

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.end - self.start,)


@register_keras_serializable()
class UnpadLayer(Layer):
    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        h_start = self.padding[0][0]
        h_end = -self.padding[0][1] if self.padding[0][1] else None
        w_start = self.padding[1][0]
        w_end = -self.padding[1][1] if self.padding[1][1] else None
        return inputs[:, h_start:h_end, w_start:w_end, :]

    def compute_output_shape(self, input_shape):
        h = input_shape[1] - self.padding[0][0] - self.padding[0][1]
        w = input_shape[2] - self.padding[1][0] - self.padding[1][1]
        return (input_shape[0], h, w, input_shape[3])

@register_keras_serializable()
class CastLayer(Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape
