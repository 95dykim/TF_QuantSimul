import tensorflow as tf

@tf.function
def round_differentiable(x):
    return x - tf.stop_gradient(x - tf.round(x))