# -*- coding: utf-8 -*-
# ResNet101，三个256全连接层，无Dropout

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')


class MyResNet101():
    def __init__(self, inputs):
        self.inputs = inputs

    def CreateMyModel(self):
        base_model = keras.applications.ResNet101(input_tensor=self.inputs, include_top=False, weights='imagenet')
        base_model.trainable = True
        for layer in base_model.layers:
            if "bn" in layer.name or 'conv5_block3' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        out1_score = keras.layers.Dense(64, name='out1_score')
        out1_output = keras.layers.Activation('softmax', name='out1')

        x = base_model.output
        x = keras.layers.GlobalAvgPool2D(name='conv5_block3_out_GAP')(x)
        x = out1_score(x)
        out = out1_output(x)
        return keras.Model(self.inputs, out)

# inputs = keras.Input(shape=(256, 256, 3), name="my_input")
# b = MyResNet101(inputs).CreateMyModel()
# b.summary()
