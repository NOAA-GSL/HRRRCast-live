import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class GatherLayer(tf.keras.layers.Layer):
    """Custom layer to replace Lambda for gathering indices"""
    def __init__(self, indices, axis=-1, **kwargs):
        super(GatherLayer, self).__init__(**kwargs)
        self.indices = indices
        self.axis = axis

    def call(self, inputs):
        return tf.gather(inputs, indices=self.indices, axis=self.axis)

    def get_config(self):
        config = super(GatherLayer, self).get_config()
        config.update({
            'indices': self.indices.tolist() if hasattr(self.indices, 'tolist') else self.indices,
            'axis': self.axis
        })
        return config

@tf.keras.utils.register_keras_serializable()
class ReduceMeanLayer(tf.keras.layers.Layer):
    """Custom layer to replace Lambda for reduce_mean"""
    def __init__(self, axis, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)

    def get_config(self):
        config = super(ReduceMeanLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@tf.keras.utils.register_keras_serializable()
class SliceLayer(tf.keras.layers.Layer):
    """Custom layer to replace Lambda for slicing"""
    def __init__(self, start_idx, end_idx, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start_idx = start_idx
        self.end_idx = end_idx

    def call(self, inputs):
        return inputs[:, :, :, self.start_idx:self.end_idx]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({
            'start_idx': self.start_idx,
            'end_idx': self.end_idx
        })
        return config

@tf.keras.utils.register_keras_serializable()
class UnpadLayer(tf.keras.layers.Layer):
    """Custom layer to replace Lambda for unpadding"""
    def __init__(self, padding, **kwargs):
        super(UnpadLayer, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        pad_top, pad_bottom = self.padding[0]
        pad_left, pad_right = self.padding[1]

        if pad_bottom == 0:
            end_h = None
        else:
            end_h = -pad_bottom

        if pad_right == 0:
            end_w = None
        else:
            end_w = -pad_right

        return inputs[:, pad_top:end_h, pad_left:end_w, :]

    def get_config(self):
        config = super(UnpadLayer, self).get_config()
        config.update({'padding': self.padding})
        return config
