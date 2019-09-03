from srcipt.layers import *
from srcipt.MiUNet import MiUNet

class small_UNet(keras.Model):
    def __init__(self, num_class=2):
        super(small_UNet, self).__init__()
        self.num_class = num_class
        self.unet0 = MiUNet(self.num_class)
        self.unet1 = MiUNet(self.num_class)
        self.unet2 = MiUNet(self.num_class)
        # self.unet3 = MiUNet(self.num_class)
        # self.unet4 = MiUNet(self.num_class)
        # self.unet5 = MiUNet(self.num_class)
        # self.unet6 = MiUNet(self.num_class)

    def call(self, inputs, **kwargs):
        o = self.unet0(inputs)

        nin0 = tf.expand_dims(o[:, :, :, -1], axis=-1)
        nin = layers.concatenate([nin0, inputs], axis=-1)
        o1 = self.unet1(nin)

        nin1 = tf.expand_dims(o1[:, :, :, -1], axis=-1)
        nin = layers.concatenate([nin0, nin1, inputs], axis=-1)
        o2 = self.unet2(nin)

        # nin2 = tf.expand_dims(o2[:, :, :, -1], axis=-1)
        # nin = layers.concatenate([nin0, nin1, nin2, inputs], axis=-1)
        # o3 = self.unet3(nin)
        #
        # nin3 = tf.expand_dims(o3[:, :, :, -1], axis=-1)
        # nin = layers.concatenate([nin0, nin1, nin2,nin3, inputs], axis=-1)
        # o4 = self.unet4(nin)

        # nin4 = tf.expand_dims(o4[:, :, :, -1], axis=-1)
        # nin = layers.concatenate([nin0, nin1, nin2, nin3,nin4, inputs], axis=-1)
        # o5 = self.unet5(nin)
        #
        # nin5 = tf.expand_dims(o5[:, :, :, -1], axis=-1)
        # nin = layers.concatenate([nin0, nin1, nin2, nin3, nin4,nin5, inputs], axis=-1)
        # o6 = self.unet6(nin)

        return o, o1, o2#, o3, o4#, o5, o6
