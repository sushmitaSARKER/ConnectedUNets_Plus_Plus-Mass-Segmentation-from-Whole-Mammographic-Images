import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Add, MaxPooling2D, Activation, Conv2DTranspose,
    BatchNormalization, GlobalAveragePooling2D, Dense, Multiply,
    UpSampling2D, concatenate
)
from tensorflow.keras import backend as K


class ConnectedUNetPlusModel:
    def __init__(self, target_size):
        self.target_size = target_size

    @staticmethod
    def conv2d_bn(x, filters, kernel_size, padding='same', strides=(1, 1), activation="relu", name=None):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=-1, scale=False)(x)
        if activation:
            x = Activation(activation, name=name)(x)
        return x

    def res_path(self, filters, length, inp):
        out = inp
        for _ in range(length):
            shortcut = self.conv2d_bn(out, filters, (1, 1), activation=None)
            out = self.conv2d_bn(out, filters, (3, 3))
            out = Add()([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization(axis=-1)(out)
        return out

    @staticmethod
    def aspp_block(x, num_filters, rate_scale=1):
        rates = [6, 12, 18]
        atrous = [
            Conv2D(num_filters, (3, 3), dilation_rate=(rate * rate_scale, rate * rate_scale), padding='same')(x)
            for rate in rates
        ]
        atrous.append(Conv2D(num_filters, (3, 3), padding='same')(x))
        atrous = [BatchNormalization()(a) for a in atrous]
        y = Add()(atrous)
        return Conv2D(num_filters, (1, 1), padding="same")(y)

    def encoder_block(self, x, filters, res_length):
        x = self.conv2d_bn(x, filters, (3, 3))
        x = self.conv2d_bn(x, filters, (3, 3))
        pooled = MaxPooling2D(pool_size=(2, 2))(x)
        res = self.res_path(filters, res_length, x)
        return pooled, res

    def decoder_block(self, x, skip, filters, res_length):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, skip], axis=-1)
        x = self.conv2d_bn(x, filters, (3, 3))
        x = self.conv2d_bn(x, filters, (3, 3))
        return self.res_path(filters, res_length, x)

    def get_model(self):
        inputs = Input((self.target_size, self.target_size, 3))

        # Encoder
        pool1, res1 = self.encoder_block(inputs, 32, 4)
        pool2, res2 = self.encoder_block(pool1, 64, 3)
        pool3, res3 = self.encoder_block(pool2, 128, 2)
        pool4, res4 = self.encoder_block(pool3, 256, 1)

        # Bottleneck
        bottleneck = self.aspp_block(pool4, 512)

        # Decoder
        up6 = self.decoder_block(bottleneck, res4, 256, 1)
        up7 = self.decoder_block(up6, res3, 128, 2)
        up8 = self.decoder_block(up7, res2, 64, 3)
        up9 = self.decoder_block(up8, res1, 32, 4)

        # Output
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(up9)

        model = Model(inputs, outputs)
        model.summary()
        return model
