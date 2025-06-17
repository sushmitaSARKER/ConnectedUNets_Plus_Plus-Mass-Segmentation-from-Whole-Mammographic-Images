import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, concatenate, Conv2D, Add, MaxPooling2D, Activation, Conv2DTranspose, 
    BatchNormalization, GlobalAveragePooling2D, Reshape, Dense, Multiply
)


class ConnectedUNetsPlusPlus:
    def __init__(self, target_size):
        self.target_size = target_size

    @staticmethod
    def aspp_block(x, num_filters, rate_scale=1):
        dilations = [6, 12, 18]
        features = [
            BatchNormalization()(Conv2D(num_filters, (3, 3), dilation_rate=(d * rate_scale, d * rate_scale), padding="same")(x))
            for d in dilations
        ]
        features.append(BatchNormalization()(Conv2D(num_filters, (3, 3), padding="same")(x)))

        y = Add()(features)
        return Conv2D(num_filters, (3, 3), padding="same")(y)

    @staticmethod
    def squeeze_excite_block(inputs, ratio=8):
        filters = inputs.shape[-1]
        se = GlobalAveragePooling2D()(inputs)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)
        return Multiply()([inputs, se])

    def resnet_block(self, x, filters, strides=1):
        shortcut = BatchNormalization()(Conv2D(filters, (1, 1), strides=strides, padding="same")(x))
        x = Activation("relu")(BatchNormalization()(Conv2D(filters, (3, 3), strides=strides, padding="same")(x)))
        x = Activation("relu")(BatchNormalization()(Conv2D(filters, (3, 3), padding="same")(x)))
        return self.squeeze_excite_block(Add()([x, shortcut]))

    @staticmethod
    def conv2d_bn(x, filters, kernel_size, padding='same', strides=(1, 1), activation='relu'):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        if activation:
            x = Activation(activation)(x)
        return x

    def res_path(self, filters, length, inp):
        for i in range(length):
            shortcut = inp if i == 0 else out
            shortcut = self.conv2d_bn(shortcut, filters, (1, 1), activation=None)
            out = self.conv2d_bn(inp if i == 0 else out, filters, (3, 3))
            out = Activation('relu')(Add()([shortcut, out]))
            out = BatchNormalization(axis=-1)(out)
        return out

    def multi_res_block(self, filters, inp, alpha=1.67):
        W = int(alpha * filters)
        shortcut = self.conv2d_bn(inp, int(W * 1.0), (1, 1), activation=None)

        conv3x3 = self.conv2d_bn(inp, int(W * 0.167), (3, 3))
        conv5x5 = self.conv2d_bn(conv3x3, int(W * 0.333), (3, 3))
        conv7x7 = self.conv2d_bn(conv5x5, int(W * 0.5), (3, 3))

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
        out = Activation('relu')(Add()([shortcut, BatchNormalization()(out)]))
        return BatchNormalization(axis=-1)(out)

    def get_rwnet(self):
        inputs = Input((self.target_size, self.target_size, 3))
        encoder_outputs = []
        filters = [32, 64, 128, 256, 512]

        # Encoder
        x = inputs
        for i, f in enumerate(filters[:-1]):
            x = self.multi_res_block(f, x)
            encoder_outputs.append(self.res_path(f, len(filters) - i - 1, x))
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Bottleneck
        x = self.aspp_block(x, filters[-1])

        # Decoder
        for i, f in reversed(list(enumerate(filters[:-1]))):
            x = concatenate([Conv2DTranspose(f, (2, 2), strides=(2, 2), padding="same")(x), encoder_outputs[i]], axis=-1)
            x = self.multi_res_block(f, x)

        outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
