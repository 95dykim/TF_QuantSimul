import tensorflow as tf
import numpy as np

from utils import round_differentiable

QUANTIZATION_LEVEL = 256.0
USE_MINOR_TWEAK = False

#Integer quantization
@tf.function
def QuantSimul(w, w_min, w_max, quant_level = QUANTIZATION_LEVEL):
    w_s = (w_max - w_min)/(quant_level-1)

    zero_nudge_int = round_differentiable( (tf.minimum(tf.maximum(0.0, w_min), w_max) - w_min) / w_s )
    if USE_MINOR_TWEAK:
        zero_nudge_int = round_differentiable( zero_nudge_int / (quant_level-1.0) * (quant_level-2.0) + 1.0 )
    zero_nudge = zero_nudge_int * w_s + w_min

    w_min = w_min - zero_nudge
    w_max = w_max - zero_nudge
    
    w_quantized_int = round_differentiable( (tf.minimum(tf.maximum(w - zero_nudge, w_min), w_max) - w_min) / w_s )
    if USE_MINOR_TWEAK:
        w_quantized_int = round_differentiable( w_quantized_int / (quant_level-1.0) * (quant_level-2.0) + 1.0 )
    w_quantized = w_quantized_int * w_s + w_min

    return w_quantized

#Conv2D Layer with quantized outputs
@tf.keras.utils.register_keras_serializable()
class QuantSimulConv2D(tf.keras.layers.Conv2D):
    def call(self, inputs):
        kernel_quantized = QuantSimul(self.kernel, tf.reduce_min(self.kernel), tf.reduce_max(self.kernel))

        result = self.convolution_op( inputs, kernel_quantized )
        if self.use_bias:
            result = result + self.bias
        return result
    
#Dense Layer with qunatized outputs
@tf.keras.utils.register_keras_serializable()
class QuantSimulDense(tf.keras.layers.Dense):
    def call(self, inputs):
        kernel_quantized = QuantSimul(self.kernel, tf.reduce_min(self.kernel), tf.reduce_max(self.kernel))

        result = tf.matmul( inputs, kernel_quantized )
        if self.use_bias:
            result = result + self.bias
        return result
    
#Quantization of activations
@tf.keras.utils.register_keras_serializable()
class QuantActivation(tf.keras.layers.Layer):
    def __init__(self, qunatization_start=0.0, **kwargs):
        super().__init__()
        
        self.smoothing = tf.Variable(0.99, name="smoothing", trainable=False)
        self.steps_total = tf.Variable(0.0 - qunatization_start, name="steps_total", trainable=False)
        self.w_min = tf.Variable(0.0, name="w_min", trainable=False)
        self.w_max = tf.Variable(0.0, name="w_max", trainable=False)
    
    def call(self, inputs, training=None):
        if training:
            if self.steps_total < 0.0:
                self.steps_total.assign_add(1.0)
                return inputs
            
            #EMA
            coeff = self.smoothing / (1.0+self.steps_total)
            self.w_min.assign( tf.reduce_min(inputs) * coeff + self.w_min * ( 1.0 - coeff ) )
            self.w_max.assign( tf.reduce_max(inputs) * coeff + self.w_max * ( 1.0 - coeff ) )
            self.steps_total.assign_add(1.0)
            
            act_quantized = QuantSimul(inputs, self.w_min, self.w_max)
            return act_quantized
        else:
            if self.steps_total < 0.0:
                return inputs
            act_quantized = QuantSimul(inputs, self.w_min, self.w_max)
            return act_quantized

    def get_config(self):
        config = super().get_config()
        return config

#BN Folding
def setFoldedWeights(model_to_convert):
    layer_stack = []
    for layer in model_to_convert.layers:
        if isinstance( layer, QuantSimulConv2D ):
            layer_stack.append( layer )
        if isinstance( layer, tf.keras.layers.BatchNormalization):
            beta = layer.beta
            gamma = layer.gamma
            epsilon = layer.epsilon
            m_mean = layer.moving_mean
            m_var = layer.moving_variance
            
            layer_to_convert = layer_stack.pop()
            if layer_to_convert.use_bias == False:
                layer_to_convert.use_bias = True
                layer_to_convert.bias = tf.Variable( np.zeros( layer_to_convert.kernel.shape[-1] ).astype(np.float32) )
            layer_to_convert.bias = layer_to_convert.bias + (beta - gamma * m_mean / m_var )
            layer_to_convert.kernel = layer_to_convert.kernel * gamma / tf.sqrt(m_var + epsilon)