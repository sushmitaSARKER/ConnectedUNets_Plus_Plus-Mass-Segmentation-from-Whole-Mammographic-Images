import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, \
              Conv2D, Add, MaxPooling2D, Activation, add, \
              Conv2DTranspose, BatchNormalization, \
              GlobalAveragePooling2D, Reshape, Dense, Multiply, \
              Lambda, UpSampling2D
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
# from tensorflow.math import multiply





class connectedunets_plus_model():
    def __init__(self, target_size):
        self.target_size = target_size


    
    def conv2d_bn(self, x, filters, num_row, num_col, padding = 'same', strides = (1,1), activation = "relu", name = None):
        x = Conv2D(filters,(num_row, num_col), strides=strides, padding = padding, use_bias = False)(x)
        x = BatchNormalization(axis=3, scale = False)(x)
        if activation == None:
            return x

        x = Activation(activation, name=name)(x)
        return x

    def ResPath(self, filters, length, inp):
        for i in range(length):
            if i == 0:
                shortcut = inp
                out = inp
            else:
                shortcut = out

            shortcut = self.conv2d_bn(shortcut, filters, 1, 1, activation = None, padding= 'same')
            out = self.conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization(axis=3)(out)

        return out

    def aspp_block(self, x, num_filters, rate_scale=1):
        x1 = Conv2D(num_filters, (3,3), dilation_rate=(6*rate_scale, 6*rate_scale), padding= 'same')(x)
        x1= BatchNormalization()(x1)

        x2 = Conv2D(num_filters, (3,3), dilation_rate=(12*rate_scale, 12*rate_scale), padding='same')(x)
        x2 = BatchNormalization()(x2)

        x3 = Conv2D(num_filters, (3,3), dilation_rate=(18*rate_scale, 18*rate_scale), padding='same')(x)
        x3 = BatchNormalization()(x3)

        x4 = Conv2D(num_filters, (3,3), padding='same')(x)
        x4 = BatchNormalization()(x4)
        
        y = Add()([x1,x2,x3,x4])
        y= Conv2D(num_filters, (1,1), padding="same")(y)
        return y


    def get_wnet(self):
        inputs = Input((self.target_size, self.target_size, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        ## Mine
        mresblock1 = self.ResPath(32, 4, conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        ## Mine
        mresblock2 = self.ResPath(32*2, 3, conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        ## Mine
        mresblock3 = self.ResPath(32*4, 2, conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        ## Mine
        mresblock4 = self.ResPath(32*8, 1, conv4)

        conv5 = self.aspp_block(pool4, 512)
    
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), mresblock4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        ## Mine
        mresblock6 = self.ResPath(32*8, 1, conv6)
        
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), mresblock3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        ## Mine
        mresblock7 = self.ResPath(32*4, 2, conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), mresblock2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
        ## Mine
        mresblock8 = self.ResPath(32*2, 3, conv8)
        
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), mresblock1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)
        
        down10 = concatenate([Conv2D(32, (3, 3), activation='relu', padding='same')(conv9), conv9], axis=3)
        conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(down10)
        conv10 = BatchNormalization()(conv10)
        conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)
        conv10 = BatchNormalization()(conv10)    
        pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
        ## Mine
        mresblock10 = self.ResPath(32, 4, conv10)

        down11 = concatenate([Conv2D(64, (3, 3), activation='relu', padding='same')(pool10), mresblock8], axis=3)
        conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(down11)
        conv11 = BatchNormalization()(conv11)
        conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)
        conv11 = BatchNormalization()(conv11)   
        pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
        ## Mine
        mresblock11 = self.ResPath(32*2, 3, conv11)
        
        down12 = concatenate([Conv2D(128, (3, 3), activation='relu', padding='same')(pool11), mresblock7], axis=3)
        conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(down12)
        conv12 = BatchNormalization()(conv12)
        conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv12)    
        conv12 = BatchNormalization()(conv12)
        pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
        ## Mine
        mresblock12 = self.ResPath(32*4, 2, conv12)

        down13 = concatenate([Conv2D(256, (3, 3), activation='relu', padding='same')(pool12), mresblock6], axis=3)
        conv13 = Conv2D(256, (3, 3), activation='relu', padding='same')(down13)
        conv13 = BatchNormalization()(conv13)
        conv13 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv13)  
        conv13 = BatchNormalization()(conv13)    
        pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
        ## Mine
        mresblock13 = self.ResPath(32*8, 1, conv13)
        
        conv14 = self.aspp_block(pool13, 512)
        
        up15 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv14), mresblock13], axis=3)
        conv15 = Conv2D(256, (3, 3), activation='relu', padding='same')(up15)
        conv15 = BatchNormalization()(conv15)    
        conv15 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv15)
        conv15 = BatchNormalization()(conv15) 
        
        up16 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv15), mresblock12], axis=3)
        conv16 = Conv2D(128, (3, 3), activation='relu', padding='same')(up16)
        conv16 = BatchNormalization()(conv16)     
        conv16 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv16)
        conv16 = BatchNormalization()(conv16)      

        up17 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv16), mresblock11], axis=3)
        conv17 = Conv2D(64, (3, 3), activation='relu', padding='same')(up17)
        conv17 = BatchNormalization()(conv17)      
        conv17 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv17)
        conv17 = BatchNormalization()(conv17)  
        
        up18 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv17), mresblock10], axis=3)
        conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(up18)
        conv18 = BatchNormalization()(conv18)      
        conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv18)
        conv18 = BatchNormalization()(conv18)    
        
        conv18 = self.aspp_block(conv18, 32)
        
        conv19 = Conv2D(1, (1, 1), activation='sigmoid')(conv18)

        model = Model(inputs=[inputs], outputs=[conv19])
        #model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef, jacard, iou_coef, 'accuracy'])
        model.summary()
        return model





