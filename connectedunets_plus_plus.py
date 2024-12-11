import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, \
    Add, MaxPooling2D, Activation, add, Conv2DTranspose, BatchNormalization, \
        GlobalAveragePooling2D, Reshape, Dense, Multiply, Lambda, UpSampling2D
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
#from tensorflow.math import multiply



class connectedunets_plus_plus_model():
    def __init__(self, target_size):
        self.target_size = target_size



    def aspp_block(self, x, num_filters, rate_scale=1):
        x1 = Conv2D(num_filters, (3,3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(num_filters, (3,3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
        x2 = BatchNormalization()(x2)

        x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
        x3 = BatchNormalization()(x3)

        x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
        x4 = BatchNormalization()(x4)

        y = Add()([x1, x2, x3, x4])
        y = Conv2D(num_filters, (3,3), padding="same")(y)
        return y



    def squeeze_excite_block(self, inputs, ratio=8):
        init = inputs
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape =(1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = Multiply()([init, se])
        return x


    def resnet_block(self, x, n_filter, strides=1):
        x_init = x

        ## Conv 1
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        ## Conv 2
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

        ## Shortcut
        s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        ## Add
        x = Add()([x, s])
        x = self.squeeze_excite_block(x)
        return x




    def conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=3, scale=False)(x)

        if(activation == None):
            return x

        x = Activation(activation, name=name)(x)

        return x

    def ResPath(self, filters, length, inp):
        '''
        ResPath
        
        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''
        for i in range(length):
            if i == 0:
                shortcut = inp
                out = inp
            else: shortcut = out

        shortcut = self.conv2d_bn(shortcut, filters, 1, 1,
                                activation=None, padding='same')

        out = self.conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

        return out


    def MultiResBlock(self, U, inp, alpha = 1.67):
        '''
        MultiRes Block
        
        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''

        W = alpha * U

        shortcut = inp

        shortcut = self.conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                            int(W*0.5), 1, 1, activation=None, padding='same')

        conv3x3 = self.conv2d_bn(inp, int(W*0.167), 3, 3,
                            activation='relu', padding='same')

        conv5x5 = self.conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                            activation='relu', padding='same')

        conv7x7 = self.conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                            activation='relu', padding='same')

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
        out = BatchNormalization(axis=3)(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

        return out

    def get_rwnet(self):

        inputs = Input((self.target_size, self.target_size, 3))
        
        mresblock1 = self.MultiResBlock(32, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
        ## Mine
        mresblock1 = self.ResPath(32, 4, mresblock1)

        mresblock2 = self.MultiResBlock(32*2, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
        ## Mine
        mresblock2 = self.ResPath(32*2, 3, mresblock2)

        mresblock3 = self.MultiResBlock(32*4, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
        ## Mine
        mresblock3 = self.ResPath(32*4, 2, mresblock3)

        mresblock4 = self.MultiResBlock(32*8, pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
        ## Mine
        mresblock4 = self.ResPath(32*8, 1, mresblock4)

        conv5 = self.aspp_block(pool4, 512)
        
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), mresblock4], axis=3)
        conv6 = self.MultiResBlock(32*8, up6)
        ## Mine
        mresblock6 = self.ResPath(32*8, 1, conv6)
        
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), mresblock3], axis=3)
        conv7 = self.MultiResBlock(32*4, up7)
        ## Mine
        mresblock7 = self.ResPath(32*4, 2, conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), mresblock2], axis=3)
        conv8 = self.MultiResBlock(32*2, up8)
        ## Mine
        mresblock8 = self.ResPath(32*2, 3, conv8)
        
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), mresblock1], axis=3)
        conv9 = self.MultiResBlock(32, up9)
        
        down10 = concatenate([Conv2D(32, (3, 3), activation='relu', padding='same')(conv9), conv9], axis=3)  
        mresblock10 = self.MultiResBlock(32, down10)  
        #conv10 = resnet_block(down10, 32, strides=1)  
        pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
        ## Mine
        mresblock10 = self.ResPath(32, 4, mresblock10)

        down11 = concatenate([Conv2D(64, (3, 3), activation='relu', padding='same')(pool10), mresblock8], axis=3)
        mresblock11 = self.MultiResBlock(32*2, down11) 
        pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
        ## Mine
        mresblock11 = self.ResPath(32*2, 3, mresblock11)
        
        down12 = concatenate([Conv2D(128, (3, 3), activation='relu', padding='same')(pool11), mresblock7], axis=3)
        mresblock12 = self.MultiResBlock(32*4, down12)
        #conv12 = resnet_block(down12, 128, strides=1)
        pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
        ## Mine
        mresblock12 = self.ResPath(32*4, 2, mresblock12)

        down13 = concatenate([Conv2D(256, (3, 3), activation='relu', padding='same')(pool12), mresblock6], axis=3)
        mresblock13 = self.MultiResBlock(32*8, down13)
        #conv13 = resnet_block(down13, 256, strides=1)
        pool13 = MaxPooling2D(pool_size=(2, 2))(mresblock13)
        ## Mine
        mresblock13 = self.ResPath(32*8, 1, mresblock13)
        
        conv14 = self.aspp_block(pool13, 512)
        
        up15 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv14), mresblock13], axis=3)
        conv15 = self.MultiResBlock(32*8, up15) 
        
        up16 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv15), mresblock12], axis=3)
        conv16 = self.MultiResBlock(32*4, up16)      

        up17 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv16), mresblock11], axis=3)
        conv17 = self.MultiResBlock(32*2, up17)    
        
        up18 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv17), mresblock10], axis=3)
        conv18 = self.MultiResBlock(32, up18)  
        
        conv18 = self.aspp_block(conv18, 32)
        
        conv19 = Conv2D(1, (1, 1), activation='sigmoid')(conv18)

        model = Model(inputs=[inputs], outputs=[conv19])
        model.summary()
        return model
  