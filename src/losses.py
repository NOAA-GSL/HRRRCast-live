import tensorflow as tf


class WeightedLoss(tf.keras.losses.Loss):
    def __init__(self, weights, **kwargs):
        super(WeightedLoss, self).__init__(**kwargs)
        self.weights = weights

    def get_config(self):
        config = super(WeightedLoss, self).get_config()
        config.update(
            {   
                "weights": (
                    self.weights
                    if isinstance(self.weights, list)
                    else self.weights.tolist()
                ),  
            }   
        )   
        return config


@tf.keras.utils.register_keras_serializable()
class WeightedMSE(WeightedLoss):
    def __init__(self, weights, **kwargs):
        super(WeightedMSE, self).__init__(weights=weights, **kwargs)

    def call(self, y_true, y_pred):
        reduction_axis = list(range(1, len(y_pred.shape)))
        return tf.reduce_mean(
            self.weights * tf.square(y_true - y_pred),
            axis=reduction_axis,
        )
