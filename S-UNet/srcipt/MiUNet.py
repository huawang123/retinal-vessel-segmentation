from srcipt.layers import *

class MiUNet(keras.Model):
    def __init__(self, num_class=2):
        super(MiUNet, self).__init__()
        self.num_class = num_class
        self.b11 = ConvBNActDouble(32, kernel_size=(3, 3), padding='same')
        self.p11 = layers.MaxPool2D()
        self.b12 = ConvBNActDouble(20, kernel_size=(3, 3), padding='same')
        self.p12 = layers.MaxPool2D()
        self.b13 = ConvBNActDouble(12, kernel_size=(3, 3), padding='same')
        self.p13 = layers.MaxPool2D()
        self.b14 = ConvBNActDouble(12, kernel_size=(3, 3), padding='same')
        self.p14 = layers.MaxPool2D()

        self.b15 = ConvBNAct(12, kernel_size=(3, 3), padding='same')

        self.d14 = DeconvConcatBNAct(12, kernel_size=(3, 3), padding='same')
        self.d14_1 = ConvBNActDouble(12, kernel_size=(3, 3), padding='same')
        self.d13 = DeconvConcatBNAct(12, kernel_size=(3, 3), padding='same')
        self.d13_1 = ConvBNActDouble(12, kernel_size=(3, 3), padding='same')
        self.d12 = DeconvConcatBNAct(20, kernel_size=(3, 3), padding='same')
        self.d12_1 = ConvBNActDouble(20, kernel_size=(3, 3), padding='same')
        self.d11 = DeconvConcatBNAct(32, kernel_size=(3, 3), padding='same')
        self.d11_1 = ConvBNActDouble(32, kernel_size=(3, 3), padding='same')
        self.d1 = ConvBNAct(32, kernel_size=(3, 3), padding='same')
        self.out1 = layers.Conv2D(self.num_class, kernel_size=(1, 1))

    def call(self, inputs, **kwargs):
        out1 = self.b11(inputs)
        down1 = self.p11(out1)
        out2 = self.b12(down1)
        down2 = self.p12(out2)
        out3 = self.b13(down2)
        down3 = self.p13(out3)
        out4 = self.b14(down3)
        down4 = self.p14(out4)

        out5 = self.b15(down4)

        out6 = self.d14_1(self.d14((out5, out4)))
        out7 = self.d13_1(self.d13((out6, out3)))
        out8 = self.d12_1(self.d12((out7, out2)))
        out9 = self.d11_1(self.d11((out8, out1)))
        o = self.out1(self.d1(out9))

        return o