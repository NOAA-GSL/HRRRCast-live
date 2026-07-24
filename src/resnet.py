import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from tensorflow.keras.layers import Layer, Conv2D
from diffusion_params import NUM_DIFFUSION_STEPS, SQRT_ALPHA_BAR

L2_REG = tf.keras.regularizers.L2(5e-5)
K_INIT = "glorot_uniform"

@register_keras_serializable()
class SpatialGroupedLayer(Layer):
    """
    Generic spatially grouped layer that applies any Keras layer to spatial tiles of the input tensor.
    Args:
        layer: Keras layer instance or callable (e.g., Conv2D, LayerNormalization, Activation).
        groups_h: Number of groups along height.
        groups_w: Number of groups along width.
        pad_h: Padding size along height (default: 0).
        pad_w: Padding size along width (default: 0).
        layer_kwargs: Additional keyword arguments for the layer.
    """
    def __init__(
        self,
        layer,
        groups_h=1,
        groups_w=1,
        pad_h=0,
        pad_w=0,
        layer_kwargs=None,
        layer_is_class=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.groups_h = groups_h
        self.groups_w = groups_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.layer_kwargs = layer_kwargs or {}
        self.layer_is_class = (
            isinstance(layer, type) if layer_is_class is None else layer_is_class
        )
        self._layer_instance = None

    def build(self, input_shape):
        # Create a single shared layer instance
        if self.layer_is_class or isinstance(self.layer, type):
            self._layer_instance = self.layer(**self.layer_kwargs)
        else:
            self._layer_instance = self.layer
        super().build(input_shape)

    def call(self, x):
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        tile_h = tf.cast(tf.math.ceil(tf.cast(H, tf.float32) / self.groups_h), tf.int32)
        tile_w = tf.cast(tf.math.ceil(tf.cast(W, tf.float32) / self.groups_w), tf.int32)

        outputs = []
        for i in range(self.groups_h):
            row_outputs = []
            for j in range(self.groups_w):
                # Compute tile boundaries with padding
                y0 = tf.maximum(0, i * tile_h - self.pad_h)
                y1 = tf.minimum(H, (i + 1) * tile_h + self.pad_h)
                x0 = tf.maximum(0, j * tile_w - self.pad_w)
                x1 = tf.minimum(W, (j + 1) * tile_w + self.pad_w)

                tile = x[:, y0:y1, x0:x1, :]
                tile_out = self._layer_instance(tile)

                # Crop overlap so tiles fit perfectly when concatenated
                crop_top = self.pad_h if i > 0 else 0
                crop_bottom = self.pad_h if i < self.groups_h - 1 else 0
                crop_left = self.pad_w if j > 0 else 0
                crop_right = self.pad_w if j < self.groups_w - 1 else 0

                if any([crop_top, crop_bottom, crop_left, crop_right]):
                    tile_out = tile_out[
                        :,
                        crop_top or 0 : tf.shape(tile_out)[1] - (crop_bottom or 0),
                        crop_left or 0 : tf.shape(tile_out)[2] - (crop_right or 0),
                        :
                    ]

                row_outputs.append(tile_out)
            outputs.append(tf.concat(row_outputs, axis=2))
        return tf.concat(outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        if self.layer_is_class or isinstance(self.layer, type):
            layer_cfg = self.layer.__name__
            layer_is_class = True
        else:
            layer_cfg = serialize_keras_object(self.layer)
            layer_is_class = False

        config.update({
            "layer": layer_cfg,
            "layer_is_class": layer_is_class,
            "groups_h": self.groups_h,
            "groups_w": self.groups_w,
            "pad_h": self.pad_h,
            "pad_w": self.pad_w,
            "layer_kwargs": self.layer_kwargs,
        })
        return config

    @classmethod
    def from_config(cls, config):
        layer_cfg = config.pop("layer")
        layer_is_class = config.pop("layer_is_class", False)

        if layer_is_class:
            layer_obj = getattr(tf.keras.layers, layer_cfg, None)
            if layer_obj is None:
                raise ValueError(f"Unknown keras layer class '{layer_cfg}' in SpatialGroupedLayer config.")
        else:
            layer_obj = deserialize_keras_object(layer_cfg)

        return cls(layer=layer_obj, layer_is_class=layer_is_class, **config)

@register_keras_serializable()
class ChannelPoolAvg(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Compute mean across channel axis, keep dims for broadcasting
        return tf.keras.backend.mean(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        # Output shape same as input except channels become 1
        return input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        return config
    
@register_keras_serializable()
class RecomputeSubModel(tf.keras.layers.Layer):
    """Gradient checkpointing wrapper for memory-efficient training.
    
    Recomputes intermediate activations during backpropagation instead of storing them,
    reducing memory usage at the cost of additional computation time.
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
    """Extracts and conditions time/ensemble information from input features.
    
    Supports multiple modes:
    - Case A: Full time conditioning (lead time + ensemble ID)
    - Case B: CRPS mode with only lead time
    - Case C: CRPS mode with noise vector + lead time
    """
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
        time_mask = tf.constant(self.time_mask, dtype=tf.int32)
        n_channels = tf.shape(inputs)[-1]
        time_mask = tf.where(time_mask < 0, time_mask + n_channels, time_mask)
        time_feats = tf.gather(inputs, time_mask, axis=-1)

        d = time_feats[:, 0, 0, :]

        # Case A: full d vector (lead time + ens_id + 6 other time features)
        if not self.use_crps:
            return d

        # Case B: CRPS without noise
        lead_time = d[:, -1:]  # (B, 1)
        extra_time_feats = d[:, :-1]
        if not self.use_noise:
            return tf.concat([extra_time_feats, lead_time], axis=1)

        # Case C: CRPS with noise
        ens_id = tf.cast(tf.floor(tf.cast(d[:, -2], tf.float32) * (2**31 - 1)), tf.int32)  # (B,)
        seed = tf.stack([ens_id, ens_id ^ 0x9E3779B9], axis=1)  # (B,2)
        z = tf.random.stateless_normal([tf.shape(d)[0], 32], seed=seed, dtype=lead_time.dtype)  # (B,32)
        return tf.concat([z, extra_time_feats, lead_time], axis=1)

    def compute_output_shape(self, input_shape):
        if not (self.use_crps and self.use_noise):
            return (input_shape[0], len(self.time_mask))     # Case A and B
        else:
            return (input_shape[0], 32 + len(self.time_mask))    # Case C

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
    """Applies reflective padding to tensors."""
    def __init__(self, padding, **kwargs):
        """
        Args:
            padding: Padding specification as [[h_top, h_bottom], [w_left, w_right]].
        """
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, self.padding, mode="REFLECT")

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] = shape[1] + self.padding[0][0] + self.padding[0][1]
        shape[2] = shape[2] + self.padding[1][0] + self.padding[1][1]
        return tuple(shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'padding': self.padding,
        })
        return config


@register_keras_serializable()
class OutputMaskLayer(Layer):
    """Selects specific output channels based on a mask."""
    def __init__(self, output_tensor_mask, **kwargs):
        """
        Args:
            output_tensor_mask: Indices of channels to extract.
        """
        super().__init__(**kwargs)
        self.output_tensor_mask = output_tensor_mask

    def call(self, inputs):
        return tf.gather(inputs, indices=self.output_tensor_mask, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (len(self.output_tensor_mask),)

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_tensor_mask': self.output_tensor_mask,
        })
        return config


@register_keras_serializable()
class ChannelSliceLayer(Layer):
    """Extracts a slice of channels from the input tensor."""
    def __init__(self, start, end, **kwargs):
        """
        Args:
            start: Starting channel index (inclusive).
            end: Ending channel index (exclusive).
        """
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[
            :, :, :, self.start : self.end,
        ]

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.end - self.start,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'start': self.start,
            'end': self.end,
        })
        return config


@register_keras_serializable()
class UnpadLayer(Layer):
    """Removes padding from a padded tensor."""
    def __init__(self, padding, **kwargs):
        """
        Args:
            padding: Padding specification to remove as [[h_top, h_bottom], [w_left, w_right]].
        """
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'padding': self.padding,
        })
        return config

@register_keras_serializable()
class CastLayer(Layer):
    """Casts input tensor to a specified data type."""
    def __init__(self, dtype, **kwargs):
        """
        Args:
            dtype: Target data type for casting.
        """
        super().__init__(**kwargs)
        self.target_dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'dtype': self.target_dtype,
        })
        return config
