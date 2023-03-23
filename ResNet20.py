import tensorflow as tf

#INIT = 'glorot_uniform'
#REG_L2 = tf.keras.regularizers.L2(1e-4)

def BasicBlock(x, channel_size, name, strides=1, kernel_init="glorot_uniform", kernel_reg = None):
    x_1 = x
    x_2 = x
    
    if strides != 1:
        #option_a
        x_2 = tf.keras.layers.MaxPool2D(1, strides=strides, padding="same", name=name+"_sc_maxpool"  )( tf.pad(x_2, ((0,0), (0,0), (0,0), (0, channel_size-x_2.shape[-1])), name=name+"_sc_optionA") )
        #option_b
        #x_2 = tf.keras.layers.Conv2D(channel_size, 1, strides=strides, padding='same', use_bias=False, name=name+"_sc_conv", kernel_initializer=INIT)(x_2)
        #x_2 = tf.keras.layers.BatchNormalization(name=name+"_sc_bn")(x_2)

    x_1 = tf.keras.layers.Conv2D(channel_size, 3, strides=strides, padding="same", use_bias=False, name=name+"_conv1", kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x_1)
    x_1 = tf.keras.layers.BatchNormalization(name=name+"_bn1")(x_1)
    x_1 = tf.keras.layers.Activation('relu', name=name+"_act1")(x_1)
    
    x_1 = tf.keras.layers.Conv2D(channel_size, 3, strides=1, padding="same", use_bias=False, name=name+"_conv2", kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x_1)
    x_1 = tf.keras.layers.BatchNormalization(name=name+"_bn2")(x_1)
    
    x = tf.keras.layers.Add(name=name+"_add")([x_1, x_2])
    x = tf.keras.layers.Activation('relu', name=name+"_act2")(x)

    return x

def ResNet20(input_shape=(32,32,3), classes=10, channel_sizes=16, kernel_init="glorot_uniform", kernel_reg=tf.keras.regularizers.L2(1e-4)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    #pre
    x = tf.keras.layers.Conv2D(channel_sizes, 3, strides=1, padding="same", use_bias=False, name='pre_conv', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x)
    x = tf.keras.layers.BatchNormalization(name='pre_bn')(x)
    x = tf.keras.layers.Activation('relu',name='pre_act')(x)

    #blocks_1
    x = BasicBlock(x, channel_sizes, "blocks_1_1", strides=1)
    for i in range(1,3):
        x = BasicBlock(x, channel_sizes, "blocks_1_"+str(i+1), strides=1)

    #blocks_2
    x = BasicBlock(x, channel_sizes*2, "blocks_2_1", strides=2)
    for i in range(1,3):
        x = BasicBlock(x, channel_sizes*2, "blocks_2_"+str(i+1), strides=1)

    #blocks_3
    x = BasicBlock(x, channel_sizes*4, "blocks_3_1", strides=2)
    for i in range(1,3):
        x = BasicBlock(x, channel_sizes*4, "blocks_3_"+str(i+1), strides=1)
    
    #pred
    x = tf.keras.layers.GlobalAveragePooling2D(name='pred_gap')(x)
    x = tf.keras.layers.Dense(classes, name='pred_dense', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x) #BIAS REGULARIZER X
    outputs = tf.keras.layers.Activation("softmax", name="pred_out")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet18')