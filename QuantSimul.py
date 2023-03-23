import tensorflow as tf
import numpy as np

from utils import round_differentiable

QUANTIZATION_LEVEL = 256.0
USE_MINOR_TWEAK = False

#BATCH_SIZE = 128
#ACT_QUANT_START = 1000000
#ACT_QUANT_START = round(ACT_QUANT_START/BATCH_SIZE)

#@tf.function
#def round_differentiable(x):
#    return x - tf.stop_gradient(x - tf.round(x))

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

@tf.keras.utils.register_keras_serializable()
class QuantSimulConv2D(tf.keras.layers.Conv2D):
    def call(self, inputs):
        kernel_quantized = QuantSimul(self.kernel, tf.reduce_min(self.kernel), tf.reduce_max(self.kernel))

        result = self.convolution_op( inputs, kernel_quantized )
        if self.use_bias:
            result = result + self.bias #Biases are not quantized! QuantSimul(self.bias, tf.reduce_min(self.bias), tf.reduce_max(self.bias), quant_level=float(2**32) )
        return result
    
@tf.keras.utils.register_keras_serializable()
class QuantSimulDense(tf.keras.layers.Dense):
    def call(self, inputs):
        kernel_quantized = QuantSimul(self.kernel, tf.reduce_min(self.kernel), tf.reduce_max(self.kernel))

        result = tf.matmul( inputs, kernel_quantized )
        if self.use_bias:
            result = result + self.bias #Biases are not quantized! QuantSimul(self.bias, tf.reduce_min(self.bias), tf.reduce_max(self.bias), quant_level=float(2**32) )
        return result
    
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
            
            #EMA... 이게 맞나 확인해야함. coeff*f(x) + (1-coeff)*f(x-1)고, coeff는 계속 변하는게 아니라 고정된거 아닌가?
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
    
def setFoldedWeights(model_to_convert):
    layer_stack = []
    for layer in model_to_convert.layers:
        if isinstance( layer, QuantSimulConv2D ):
            layer_stack.append( layer )
        if isinstance( layer, tf.keras.layers.BatchNormalization):
            beta = layer.beta
            gamma = layer.gamma
            epsilon = 0.0000001 #EPS 이게 맞는지 확인
            m_mean = layer.moving_mean
            m_var = layer.moving_variance
            
            #while( len(layer_stack) ): 한번만 수행
            layer_to_convert = layer_stack.pop()
            if layer_to_convert.use_bias == False:
                layer_to_convert.use_bias = True
                layer_to_convert.bias = tf.Variable( np.zeros( layer_to_convert.kernel.shape[-1] ).astype(np.float32) )
            layer_to_convert.bias = layer_to_convert.bias + (beta - gamma * m_mean / m_var )
            layer_to_convert.kernel = layer_to_convert.kernel * gamma / tf.sqrt(m_var + epsilon)