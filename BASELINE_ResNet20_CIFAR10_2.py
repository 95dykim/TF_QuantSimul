import tensorflow as tf

import os
import numpy as np
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import pandas as pd

INIT = 'glorot_uniform'
REG_L2 = tf.keras.regularizers.L2(1e-4)

def BasicBlock(x, channel_size, name, strides=1):
    x_1 = x
    x_2 = x
    
    if strides != 1:
        x_2 = tf.keras.layers.MaxPool2D(1, strides=strides, padding="same", name=name+"_sc_maxpool"  )( tf.pad(x_2, ((0,0), (0,0), (0,0), (0, channel_size-x_2.shape[-1])), name=name+"_sc_optionA") )

        #x_2 = tf.keras.layers.Conv2D(channel_size, 1, strides=strides, padding='same', use_bias=False, name=name+"_sc_conv", kernel_initializer=INIT)(x_2)
        #x_2 = tf.keras.layers.BatchNormalization(name=name+"_sc_bn")(x_2)

    x_1 = tf.keras.layers.Conv2D(channel_size, 3, strides=strides, padding="same", use_bias=False, name=name+"_conv1", kernel_initializer=INIT, kernel_regularizer=REG_L2)(x_1)
    x_1 = tf.keras.layers.BatchNormalization(name=name+"_bn1")(x_1)
    x_1 = tf.keras.layers.Activation('relu', name=name+"_act1")(x_1)
    
    x_1 = tf.keras.layers.Conv2D(channel_size, 3, strides=1, padding="same", use_bias=False, name=name+"_conv2", kernel_initializer=INIT, kernel_regularizer=REG_L2)(x_1)
    x_1 = tf.keras.layers.BatchNormalization(name=name+"_bn2")(x_1)
    
    x = tf.keras.layers.Add(name=name+"_add")([x_1, x_2])
    x = tf.keras.layers.Activation('relu', name=name+"_act2")(x)

    return x

def ResNet_20(x_mean=0.0, x_std=1.0, input_shape=(32,32,3), classes=10, channel_sizes=16):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    """
    preprocessing
    """
    #x = tf.keras.layers.RandomFlip( mode="horizontal" )(x)
    #x = tf.keras.layers.ZeroPadding2D(4)(x)
    #x = tf.keras.layers.RandomCrop( input_shape[0], input_shape[1] )(x)
    #x = tf.keras.layers.Normalization(mean=x_mean, variance=x_std)(x)
    
    """
    pre
    """
    x = tf.keras.layers.Conv2D(channel_sizes, 3, strides=1, padding="same", use_bias=False, name='pre_conv', kernel_initializer=INIT, kernel_regularizer=REG_L2)(x) #strides 2->1
    x = tf.keras.layers.BatchNormalization(name='pre_bn')(x)
    x = tf.keras.layers.Activation('relu',name='pre_act')(x)
    #x = tf.keras.layers.MaxPool2D(2, padding="same")(x) #activation map beomes too smaller with MaxPool2D
    
    """
    blocks_1
    """
    
    x = BasicBlock(x, channel_sizes, "blocks_1_1", strides=1)
    for i in range(1,3):
        x = BasicBlock(x, channel_sizes, "blocks_1_"+str(i+1), strides=1)
    
    """
    blocks_2
    """
    
    x = BasicBlock(x, channel_sizes*2, "blocks_2_1", strides=2)
    for i in range(1,3):
        x = BasicBlock(x, channel_sizes*2, "blocks_2_"+str(i+1), strides=1)
        
    """
    blocks_3
    """
    
    x = BasicBlock(x, channel_sizes*4, "blocks_3_1", strides=2)
    for i in range(1,3):
        x = BasicBlock(x, channel_sizes*4, "blocks_3_"+str(i+1), strides=1)
        
    """
    blocks_4
    """
    #x = BasicBlock(x, channel_sizes*8, "blocks_4_1", strides=2)
    #for i in range(1,2):
        #x = BasicBlock(x, channel_sizes*8, "blocks_4_"+str(i+1), strides=1)
    
    x = tf.keras.layers.GlobalAveragePooling2D(name='pred_gap')(x)
    x = tf.keras.layers.Dense(classes, name='pred_dense', kernel_initializer=INIT, kernel_regularizer=REG_L2)(x) #BIAS REGULARIZER X
    outputs = tf.keras.layers.Activation("softmax", name="pred_out")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet18')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train)# tf.squeeze( tf.one_hot( y_train, y_train.max()+1 ) )
y_test = tf.keras.utils.to_categorical(y_test) #tf.squeeze( tf.one_hot( y_test, y_test.max()+1 ) )

x_train = x_train/255.0
x_test = x_test/255.0
x_mean = np.mean(x_train, axis=(0,1,2), keepdims=True)
x_std = np.std(x_train, axis=(0,1,2), keepdims=True)

x_train = (x_train - x_mean)/x_std
x_test = (x_test - x_mean)/x_std

model = ResNet_20(x_mean, x_std)
model.summary()

#LR = 0.1
#LR_Scheduler = tf.keras.callbacks.LearningRateScheduler( tf.keras.optimizers.schedules.CosineDecayRestarts(LR, 20) ) # 300 epochs
def scheduler(epoch, lr):
    if epoch <150:
        return 0.1
    elif epoch <225:
        return 0.01
    else:
        return 0.001
def scheduler_200(epoch, lr):
    if epoch <60:
        return 0.1
    elif epoch <120:
        return 0.02
    elif epoch <160:
        return 0.004
    else:
        return 0.0008
LR_Scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

savename = "BASELINE_ResNet20_CIFAR10_2"
checkpoint_filepath = './' + savename + '/checkpoint-{epoch}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,save_best_only=False)

optim = tf.keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True)
loss_f = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
model.compile(optimizer=optim, loss=loss_f, metrics=['accuracy'])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True)

datagen.fit(x_train)
history = model.fit(datagen.flow(x_train, y_train, batch_size=128), validation_data=(x_test, y_test), epochs=300, callbacks=[model_checkpoint_callback, LR_Scheduler])

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('{}.csv'.format(savename), index=False)