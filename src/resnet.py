import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from tensorflow.keras.layers import Layer

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

    @tf.function(jit_compile=False)
    def call(self, inputs):
        return tf.recompute_grad(lambda: self.submodel(inputs))()

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
        def per_sample_fn(sample):
            time_feats = tf.gather(sample, self.time_mask, axis=-1)  # (H, W, 2)
            d = tf.reduce_mean(time_feats, axis=[0, 1])  # (2,)
            
            if not self.use_crps:
                return d  # Case A: full d vector (lead time + ens_id)
            
            lead_time = d[-1:]  # (1,)
            if not self.use_noise:
                return lead_time  # Case B: only lead_time
            
            # Case C: CRPS + noise
            ens_id = tf.cast(d[0] * 100, tf.int32)  # scalar int32
            seed = tf.stack([ens_id, ens_id ^ 0x9E3779B9])  # (2,)
            z = tf.random.stateless_normal([32], seed=seed)  # (32,)
            return tf.concat([z, lead_time], axis=0)  # (33,)

        return tf.map_fn(per_sample_fn, inputs)

    def compute_output_shape(self, input_shape):
        if not self.use_crps:
            return (input_shape[0], 2)     # Case A
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
