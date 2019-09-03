import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python import keras

class ConvBNAct(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(ConvBNAct, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size=kernel_size, **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, x, **kwargs):
        return self.act(self.bn(self.conv(x)))


class ConvBNActDouble(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(ConvBNActDouble, self).__init__()
        self.conv1 = ConvBNAct(filters, kernel_size=kernel_size, **kwargs)
        self.conv2 = ConvBNAct(filters, kernel_size=kernel_size, **kwargs)

    def call(self, x, **kwargs):
        return self.conv2(self.conv1(x))


class DeconvConcatBNAct(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(DeconvConcatBNAct, self).__init__()
        self.deconv = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2, 2), **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x, sub = inputs
        expand = self.deconv(x)
        o = tf.concat([expand, sub], axis=-1)
        return self.act(self.bn(o))

class DeconvConcatBNAct2(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(DeconvConcatBNAct2, self).__init__()
        self.deconv = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2, 2), **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x, x1, x2 = inputs
        expand = self.deconv(x)
        o = tf.concat([expand, x1, x2], axis=-1)
        return self.act(self.bn(o))

class DeconvConcatBNAct3(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(DeconvConcatBNAct3, self).__init__()
        self.deconv = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2, 2), **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x, x1, x2, x3 = inputs
        expand = self.deconv(x)
        o = tf.concat([expand, x1, x2, x3], axis=-1)
        return self.act(self.bn(o))

class DeconvConcatBNAct4(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(DeconvConcatBNAct4, self).__init__()
        self.deconv = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2, 2), **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x, x1, x2, x3, x4 = inputs
        expand = self.deconv(x)
        o = tf.concat([expand, x1, x2, x3, x4], axis=-1)
        return self.act(self.bn(o))

class Dilation_block(keras.Model):

    def __init__(self, filters: int, kernel_size: (int, int), **kwargs):
        super(Dilation_block, self).__init__()
        self.conv1 = ConvBNAct(filters, kernel_size=kernel_size, dilation_rate=(1,1), **kwargs)
        self.conv2 = ConvBNAct(filters, kernel_size=kernel_size, dilation_rate=(2,2), **kwargs)
        self.conv3 = ConvBNAct(filters, kernel_size=kernel_size, dilation_rate=(4,4), **kwargs)
        self.conv4 = ConvBNAct(filters, kernel_size=kernel_size, dilation_rate=(8,8), **kwargs)

    def call(self, x, **kwargs):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x + x1 + x2 + x3 + x4

class NIN(keras.Model):

    def __init__(self, filter: int, kernel_size: (int, int), **kwargs):
        super(NIN, self).__init__()
        self.cross_conv = layers.Conv2D(filter, kernel_size=kernel_size, **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        # x, y = inputs
        # o = tf.concat([x,y], axis=-1)
        conv = self.cross_conv(inputs)
        bn = self.bn(conv)
        act = self.act(bn)
        return act

class UnetGatingSignal(keras.Model):

    def __init__(self, filter: int, kernel_size: (int, int), **kwargs):
        super(UnetGatingSignal, self).__init__()
        self.cross_conv = layers.Conv2D(filter, kernel_size=kernel_size, **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, x, **kwargs):
        return x

class AttnGatingBlock(keras.Model):

    def __init__(self,filter:int, **kwargs):
        super(AttnGatingBlock, self).__init__()
        self.theta_x = layers.Conv2D(filter, kernel_size=(2,2), strides=(2, 2), **kwargs)
        self.phi_g = layers.Conv2D(filter, kernel_size=(1,1), **kwargs)
        self.deconv_g = layers.Conv2DTranspose(filter, kernel_size=(3,3), strides=(1,1), **kwargs)

        self.act_xg = layers.Activation('relu')
        self.conv = layers.Conv2D(1, kernel_size=(1,1), **kwargs)
        self.act_sigmoid = layers.Activation('sigmoid')
        self.upsample = layers.UpSampling2D()
        self.conv_end = layers.Conv2D(filter, kernel_size=(1, 1), **kwargs)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        g, x = inputs#128:64
        theta_x = self.theta_x(x)#64*64*12

        phi_g = self.phi_g(g)#64*64*12
        upsample_g = self.deconv_g(phi_g)

        add_xg = layers.add([upsample_g, theta_x])
        act_xg = self.act_xg(add_xg)
        psi = self.conv(act_xg)
        sigmoid_psi = self.act_sigmoid(psi)
        upsample_psi = self.upsample(sigmoid_psi)

        shape_x = x.get_shape().as_list()[-1]
        repeat_psi = tf.tile(upsample_psi, multiples=tf.constant([1,1,1,shape_x]))

        y = layers.multiply([repeat_psi, x])

        res = self.conv_end(y)
        res_bn = self.bn(res)
        return res_bn

def count_trainable_params():
    total_parameters = 0
    a = []
    for variable in tf.trainable_variables():
        a.append(variable)
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))