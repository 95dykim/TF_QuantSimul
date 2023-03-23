import tensorflow as tf
import numpy as np

from utils import round_differentiable

def get_LUT():
    def finder( maxlen, cur = [], found = [] ):
        if len(cur) == maxlen:
            found = found + [ cur ]
            return found
        else:
            found = finder( maxlen, cur + [1], found )
            found = finder( maxlen, cur + [-1], found )
            return found

    DEPTH1 = np.asarray([1.0]).reshape(1,-1)
    DEPTH2 = np.asarray([1.009, 1.591]).reshape(1,-1)
    DEPTH3 = np.asarray([0.832, 1.514, 1.897]).reshape(1,-1)
    DEPTH4 = np.asarray([0.838, 1.324, 1.619, 1.879]).reshape(1,-1)
    
    list_return = []
    for i in range(1,5):
        list_return.append( np.sort( np.sum( np.asarray(finder(i)) * locals()["DEPTH"+str(i)], axis=1) ) )
        
    return tuple( elem.tolist() for elem in list_return )

def get_edges( x ):
    return [ (x[idx]+x[idx+1])/2 for idx in range(len(x)-1) ]

LUT_1, LUT_2, LUT_3, LUT_4 = get_LUT()
LUT_EDGE_1 = get_edges( LUT_1 )
LUT_EDGE_2 = get_edges( LUT_2 )
LUT_EDGE_3 = get_edges( LUT_3 )
LUT_EDGE_4 = get_edges( LUT_4 )

TYPE_W_CONV = 1
TYPE_W_DENSE = 2

#@tf.function
#def round_differentiable(x):
#    return x - tf.stop_gradient(x - tf.round(x))

@tf.function
def MBQuant( x, vec_bitwidth, l_type ):
    if l_type == TYPE_W_CONV:
        axis = (0,1,2)
        vec_bit_rs_shape = (1,1,1,vec_bitwidth.shape[0],5)
        x_quant_shape = (1,1,1,-1)
    elif l_type == TYPE_W_DENSE:
        axis = (0,)
        vec_bit_rs_shape = (1,vec_bitwidth.shape[0],5)
        x_quant_shape = (1,-1)
    
    x_mean = tf.reduce_mean( x, axis=axis, keepdims=True )
    x_mad = tf.reduce_mean( tf.abs( x - x_mean ), axis=axis, keepdims=True )
    
    # x_mad에 미세한 값을 추가해서 divide by zero 방지
    x_norm = (x - x_mean)/(x_mad + 0.0000001)
    
    bq_0 = x_norm*0.0
    bq_1 = tf.gather( LUT_1, tf.raw_ops.Bucketize(input=x_norm, boundaries=LUT_EDGE_1) )
    bq_2 = tf.gather( LUT_2, tf.raw_ops.Bucketize(input=x_norm, boundaries=LUT_EDGE_2) )
    bq_3 = tf.gather( LUT_3, tf.raw_ops.Bucketize(input=x_norm, boundaries=LUT_EDGE_3) )
    bq_4 = tf.gather( LUT_4, tf.raw_ops.Bucketize(input=x_norm, boundaries=LUT_EDGE_4) )

    bq_stacked = tf.stack([bq_0, bq_1, bq_2, bq_3, bq_4], axis=-1)
    vec_bitwidth_rs = tf.reshape( tf.one_hot(vec_bitwidth,5, dtype=tf.float32), vec_bit_rs_shape )

    x_quant = tf.reduce_sum( vec_bitwidth_rs * bq_stacked, axis=-1 )
    
    x_quant = x_quant * x_mad + x_mean
    x_quant = ( 1.0 - tf.cast( tf.reshape( tf.equal(vec_bitwidth,0), x_quant_shape ), tf.float32 ) ) * x_quant
    
    return x_quant

@tf.function
def MBQuant_uniform( x, vec_bitwidth, tau ):
    q_level = tf.cast( 2**vec_bitwidth-1, tf.float32 )
    act_quantized = round_differentiable( tf.minimum( tf.maximum( x/tau, 0.0 ), 1.0 ) * q_level )
    return act_quantized * tau / q_level

@tf.keras.utils.register_keras_serializable()
class MBQuantSimulConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, tau=1.0, qconfig="distribution_aware", *args, **kwargs):
        super().__init__(filters,kernel_size, **kwargs)
        self.qconfig = qconfig
        self.tau = tau            
        
    def build(self, input_shape):
        super().build(input_shape)

        if self.qconfig == "distribution_aware":
            self.vec_bitwidth = tf.Variable( [4]*self.kernel.shape[-1], name="BitWidth", trainable=False, dtype=tf.int32 )
            self.quant_op = MBQuant
        elif self.qconfig == "uniform":
            self.vec_bitwidth = tf.Variable( [8]*self.kernel.shape[-1], name="BitWidth", trainable=False, dtype=tf.int32 )
            self.quant_op = MBQuant_uniform
            self.tau = tf.Variable( self.tau, name="tau", trainable=True, dtype=tf.float32 )

        self.quant_before = tf.Variable( tf.zeros( self.kernel.shape, dtype=tf.float32 ), name="qaunt_before", trainable=False )
        self.quant_after = tf.Variable( tf.zeros( self.kernel.shape, dtype=tf.float32 ), name="quant_after", trainable=False )
        
    def call(self, inputs, training=None):
        kernel_quant = self.quant_op(self.kernel, self.vec_bitwidth, self.tau if self.quant_op == MBQuant_uniform else TYPE_W_DENSE)
        
        if training:
            self.quant_before.assign( self.kernel )
            self.quant_after.assign( kernel_quant )
        
        result = self.convolution_op( inputs, kernel_quant )
        if self.use_bias:
            result = result + self.bias * tf.cast( tf.not_equal( self.vec_bitwidth, 0 ), tf.float32 )
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "qconfig": self.qconfig,
        })
        return config

    
@tf.keras.utils.register_keras_serializable()
class MBQuantDense(tf.keras.layers.Dense):
    def __init__(self, units, tau=1.0, qconfig="uniform", **kwargs):
        super().__init__(units, **kwargs)
        self.qconfig = qconfig
        self.tau = tau
        
    def build(self, input_shape):
        super().build(input_shape)
        
        if self.qconfig == "distribution_aware":
            self.vec_bitwidth = tf.Variable( [4]*self.kernel.shape[-1], name="BitWidth", trainable=False, dtype=tf.int32 )
            self.quant_op = MBQuant
        elif self.qconfig == "uniform":
            self.vec_bitwidth = tf.Variable( [8]*self.kernel.shape[-1], name="BitWidth", trainable=False, dtype=tf.int32 )
            self.quant_op = MBQuant_uniform
            self.tau = tf.Variable( self.tau, name="tau", trainable=True, dtype=tf.float32 )

        self.quant_before = tf.Variable( tf.zeros( self.kernel.shape, dtype=tf.float32 ), name="qaunt_before", trainable=False )
        self.quant_after = tf.Variable( tf.zeros( self.kernel.shape, dtype=tf.float32 ), name="quant_after", trainable=False )
    
    def call(self, inputs, training=None):
        kernel_quant = self.quant_op(self.kernel, self.vec_bitwidth, self.tau if self.quant_op == MBQuant_uniform else TYPE_W_DENSE)
        
        if training:
            self.quant_before.assign( self.kernel )
            self.quant_after.assign( kernel_quant )
        
        result = tf.matmul( inputs, kernel_quant )
        if self.use_bias:
            result = result + self.bias * tf.cast( tf.not_equal( self.vec_bitwidth, 0 ), tf.float32 )
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "qconfig": self.qconfig,
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class MBQuantActivation(tf.keras.layers.Layer):
    def __init__(self, tau=1.0, name=None, **kwargs):
        super().__init__(name=name)
        self.tau = tau
        self.qconfig = "uniform"
        
    def build(self, input_shape):
        super().build(input_shape)
        self.vec_bitwidth = tf.Variable( [4]*input_shape[-1], name="BitWidth", trainable=False, dtype=tf.int32 )
        self.tau = tf.Variable( self.tau, name="tau", trainable=True, dtype=tf.float32 )
        self.grad_catcher = tf.Variable( tf.zeros( (1,) + input_shape[1:], dtype=tf.float32 ), name="GradCatcher", trainable=True )
        
        self.quant_before = tf.Variable( tf.zeros( (1,) + input_shape[1:], dtype=tf.float32 ), name="qaunt_before", trainable=False )
        self.quant_after = tf.Variable( tf.zeros( (1,) + input_shape[1:], dtype=tf.float32 ), name="quant_after", trainable=False )
    
    def call(self, inputs, training=None):
        act_quant = MBQuant_uniform(inputs  + self.grad_catcher, self.vec_bitwidth, self.tau)
        
        if training:
            self.quant_before.assign( tf.reduce_mean(inputs, axis=0, keepdims=True) )
            self.quant_after.assign( tf.reduce_mean(act_quant, axis=0, keepdims=True) )
        
        return act_quant

    def get_config(self):
        config = super().get_config()
        config.update({
            "qconfig": self.qconfig,
        })
        return config
    
class MBQuantModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.accumulate_grads = True
        
        dict_layername2layeridx = dict()
        for idx,layer in enumerate(self.layers):
            dict_layername2layeridx[layer.name] = idx
        
        self.activation_indices = []
        self.activation_layer_indices = []
        self.activation_accumulator = []
        for idx, w in enumerate(self.trainable_weights):
            layer_name = w.name.split("/")[0]
            weight_name = w.name.split("/")[1].split(":")[0]
            
            if weight_name == "GradCatcher":
                self.activation_indices.append( idx )
                self.activation_layer_indices.append( dict_layername2layeridx[layer_name] )
                self.activation_accumulator.append( tf.Variable( tf.zeros( w.shape ), trainable=False, name="accumulated_activation" ) )

        self.weight_indices = []
        self.weight_layer_indices = []
        self.weight_accumulator = []
        for idx, w in enumerate(self.trainable_weights):
            layer_name = w.name.split("/")[0]
            weight_name = w.name.split("/")[1].split(":")[0]
            
            if weight_name == "kernel":
                if type(self.get_layer(layer_name)) == MBQuantSimulConv2D and self.get_layer(layer_name).qconfig != "uniform":
                    self.weight_indices.append( idx )
                    self.weight_layer_indices.append( dict_layername2layeridx[layer_name] )
                    self.weight_accumulator.append( tf.Variable( tf.zeros( w.shape ), trainable=False, name="accumulated_weight" ) )
                    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:                
            pred = self(x, training=True)
            loss = self.compiled_loss(y, pred)

        trainable_weights = self.trainable_weights
        
        grads = tape.gradient(loss, trainable_weights)
        
        if self.accumulate_grads:
            for idx in range(len(self.activation_indices)):
                act_idx = self.activation_indices[idx]
                lyer_idx = self.activation_layer_indices[idx]
                self.activation_accumulator[idx].assign_add( grads[act_idx] * ( self.layers[lyer_idx].quant_before - self.layers[lyer_idx].quant_after ) )
            for idx in range(len(self.weight_indices)):
                wt_idx = self.weight_indices[idx]
                lyer_idx = self.weight_layer_indices[idx]
                self.weight_accumulator[idx].assign_add( grads[wt_idx] * ( self.layers[lyer_idx].quant_before - self.layers[lyer_idx].quant_after ) )   
        
        self.optimizer.apply_gradients(zip(grads, trainable_weights))
        self.compiled_metrics.update_state(y, pred)
        
        #reset GradCatchers
        for layer in self.layers:
            if type(layer) == MBQuantActivation:
                layer.grad_catcher.assign( tf.zeros(layer.grad_catcher.shape, dtype=tf.float32) )
        
        return {m.name: m.result() for m in self.metrics}
    
class Callback_AdjustBitWidths(tf.keras.callbacks.Callback):
    def __init__(self, ratio_bitdecrease=0.15):
        super().__init__()
        self.ratio_bitdecrease = ratio_bitdecrease
    
    def on_epoch_begin(self, epoch, logs=None):
        for v in self.model.activation_accumulator:
            v.assign( tf.zeros( v.shape ) )
        for v in self.model.weight_accumulator:
            v.assign( tf.zeros( v.shape ) )

    def on_epoch_end(self, epoch, logs=None):
        list_key_grad = []
        list_key_lyer = []
        list_key_bitwidth_idx = []
        list_key_bitwidth = []

        for idx in range(len( self.model.weight_indices )):
            idx_lyer = self.model.weight_indices[idx]

            key_grad = abs( np.sum( self.model.weight_accumulator[idx], axis=(0,1,2) ) / np.product( self.model.weight_accumulator[idx].shape[:2] ) )
            key_lyer = np.ones( len(key_grad) ) * self.model.weight_indices[idx]
            key_bitwidth_idx = np.arange( len(key_grad) )
            key_bitwidth = self.model.layers[self.model.weight_indices[idx]].vec_bitwidth.numpy()

            list_key_grad.append( key_grad )
            list_key_lyer.append( key_lyer.astype(int) )
            list_key_bitwidth_idx.append( key_bitwidth_idx )
            list_key_bitwidth.append( key_bitwidth )

        list_key_grad = np.concatenate(list_key_grad)
        list_key_lyer = np.concatenate(list_key_lyer)
        list_key_bitwidth_idx = np.concatenate(list_key_bitwidth_idx)
        list_key_bitwidth = np.concatenate(list_key_bitwidth)

        sorted_g = sorted( [ elem for elem in zip(list_key_grad, list_key_lyer, list_key_bitwidth_idx, list_key_bitwidth) ] )
        cnt = int(len(sorted_g)*self.ratio_bitdecrease)
        for idx, elem in enumerate( sorted_g ):
            if elem[-1] == 0:
                continue

            self.model.layers[ elem[1] ].vec_bitwidth[ elem[2] ].assign( elem[3] -1 )
            cnt = cnt - 1

            if cnt == 0:
                break
                
        tf.print( sorted_g )