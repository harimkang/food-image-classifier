"""
Model naming and structure follows TF-slim implementation
# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)
- https://sike6054.github.io/blog/paper/fourth-post/
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    MaxPooling2D,
    Concatenate,
    AveragePooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Input,
)

from model_util import conv2d_bn

# layers = None
# backend = None


def Stem(input_tensor, version=None, name=None):
    x = conv2d_bn(
        input_tensor, 32, (3, 3), padding="valid", strides=2
    )  # 299x299x3 -> 149x149x32
    x = conv2d_bn(x, 32, (3, 3), padding="valid")  # 149x149x32 -> 147x147x32
    x = conv2d_bn(x, 64, (3, 3))  # 147x147x32 -> 147x147x64

    branch_1 = MaxPooling2D((3, 3), padding="valid", strides=2)(x)
    branch_2 = conv2d_bn(x, 96, (3, 3), padding="valid", strides=2)
    x = Concatenate()([branch_1, branch_2])  # 73x73x160

    branch_1 = conv2d_bn(x, 64, (1, 1))
    branch_1 = conv2d_bn(branch_1, 96, (3, 3), padding="valid")
    branch_2 = conv2d_bn(x, 64, (1, 1))
    branch_2 = conv2d_bn(branch_2, 64, (7, 1))
    branch_2 = conv2d_bn(branch_2, 64, (1, 7))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3), padding="valid")
    x = Concatenate()([branch_1, branch_2])  # 71x71x192

    branch_1 = conv2d_bn(x, 192, (3, 3), padding="valid", strides=2)  # Fig.4 is wrong
    branch_2 = MaxPooling2D((3, 3), padding="valid", strides=2)(x)
    x = (
        Concatenate(name=name)([branch_1, branch_2])
        if name
        else Concatenate()([branch_1, branch_2])
    )  # 35x35x384

    return x


def Inception_A(input_tensor, name=None):
    branch_1 = AveragePooling2D((3, 3), strides=1, padding="same")(input_tensor)
    branch_1 = conv2d_bn(branch_1, 96, (1, 1))

    branch_2 = conv2d_bn(input_tensor, 96, (1, 1))

    branch_3 = conv2d_bn(input_tensor, 64, (1, 1))
    branch_3 = conv2d_bn(branch_3, 96, (3, 3))

    branch_4 = conv2d_bn(input_tensor, 64, (1, 1))
    branch_4 = conv2d_bn(branch_4, 96, (3, 3))
    branch_4 = conv2d_bn(branch_4, 96, (3, 3))

    filter_concat = (
        Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4])
        if name
        else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    )

    return filter_concat


def Inception_B(input_tensor, name=None):
    branch_1 = AveragePooling2D((3, 3), strides=1, padding="same")(input_tensor)
    branch_1 = conv2d_bn(branch_1, 128, (1, 1))

    branch_2 = conv2d_bn(input_tensor, 384, (1, 1))

    branch_3 = conv2d_bn(input_tensor, 192, (1, 1))
    branch_3 = conv2d_bn(branch_3, 224, (1, 7))
    branch_3 = conv2d_bn(branch_3, 256, (7, 1))  # Fig.6 is wrong

    branch_4 = conv2d_bn(input_tensor, 192, (1, 1))
    branch_4 = conv2d_bn(branch_4, 192, (1, 7))
    branch_4 = conv2d_bn(branch_4, 224, (7, 1))
    branch_4 = conv2d_bn(branch_4, 224, (1, 7))
    branch_4 = conv2d_bn(branch_4, 256, (7, 1))

    filter_concat = (
        Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4])
        if name
        else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    )

    return filter_concat


def Inception_C(input_tensor, name=None):
    branch_1 = AveragePooling2D((3, 3), strides=1, padding="same")(input_tensor)
    branch_1 = conv2d_bn(branch_1, 256, (1, 1))

    branch_2 = conv2d_bn(input_tensor, 256, (1, 1))

    branch_3 = conv2d_bn(input_tensor, 384, (1, 1))
    branch_3a = conv2d_bn(branch_3, 256, (1, 3))
    branch_3b = conv2d_bn(branch_3, 256, (3, 1))
    branch_3 = Concatenate()([branch_3a, branch_3b])

    branch_4 = conv2d_bn(input_tensor, 384, (1, 1))
    branch_4 = conv2d_bn(branch_4, 448, (1, 3))
    branch_4 = conv2d_bn(branch_4, 512, (3, 1))
    branch_4a = conv2d_bn(branch_4, 256, (1, 3))
    branch_4b = conv2d_bn(branch_4, 256, (3, 1))
    branch_4 = Concatenate()([branch_4a, branch_4b])

    filter_concat = (
        Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4])
        if name
        else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    )

    return filter_concat


reduction_table = {"Inception-v4": [192, 224, 256, 384]}


def Reduction_A(input_tensor, version=None, name=None):
    k, l, m, n = reduction_table[version]

    branch_1 = MaxPooling2D((3, 3), padding="valid", strides=2)(input_tensor)

    branch_2 = conv2d_bn(input_tensor, n, (3, 3), padding="valid", strides=2)

    branch_3 = conv2d_bn(input_tensor, k, (1, 1))
    branch_3 = conv2d_bn(branch_3, l, (3, 3))
    branch_3 = conv2d_bn(branch_3, m, (3, 3), padding="valid", strides=2)

    filter_concat = (
        Concatenate(name=name)([branch_1, branch_2, branch_3])
        if name
        else Concatenate()([branch_1, branch_2, branch_3])
    )

    return filter_concat


def Reduction_B(input_tensor, version=None, name=None):
    branch_1 = MaxPooling2D((3, 3), padding="valid", strides=2)(input_tensor)

    branch_2 = conv2d_bn(input_tensor, 192, (1, 1))
    branch_2 = conv2d_bn(branch_2, 192, (3, 3), padding="valid", strides=2)

    branch_3 = conv2d_bn(input_tensor, 256, (1, 1))
    branch_3 = conv2d_bn(branch_3, 256, (1, 7))
    branch_3 = conv2d_bn(branch_3, 320, (7, 1))
    branch_3 = conv2d_bn(branch_3, 320, (3, 3), padding="valid", strides=2)

    filter_concat = (
        Concatenate(name=name)([branch_1, branch_2, branch_3])
        if name
        else Concatenate()([branch_1, branch_2, branch_3])
    )

    return filter_concat


def Inception_v4(input_shape, classes=1000):
    version = "Inception-v4"

    x = Stem(
        Input(input_shape), version=version, name="Stem"
    )  # (299, 299, 3) -> (35, 35, 384)

    for i in range(4):
        x = Inception_A(x, name="Inception-A-" + str(i + 1))  # (35, 35, 384)

    x = Reduction_A(
        x, version=version, name="Reduction-A"
    )  # (35, 35, 384) -> (17, 17, 1024)

    for i in range(7):
        x = Inception_B(x, name="Inception-B-" + str(i + 1))  # (17, 17, 1024)

    x = Reduction_B(
        x, version=version, name="Reduction-B"
    )  # (17, 17, 1024) -> (8, 8, 1536)

    for i in range(3):
        x = Inception_C(x, name="Inception-C-" + str(i + 1))  # (8, 8, 1536)

    x = GlobalAveragePooling2D()(x)  # (1536)
    x = Dropout(0.8)(x)

    model_output = Dense(classes, activation="softmax", name="output")(x)

    model = Model(Input(input_shape), model_output, name="Inception-v4")

    return model
