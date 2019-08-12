import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers as nn
from thundernet.utils.common import conv1x1, depthwise_conv5x5, conv1x1_block, conv3x3_block, maxpool2d, \
                                    channel_shuffle_lambda, se_block, batchnorm, is_channels_first, get_channel_axis


def context_enhancement_module(x1, x2, x3, size, name='cem_block'):
    x1 = conv1x1(x1,
                 in_channels=x1.shape[3],
                 out_channels=245,
                 strides=1,
                 groups=1,
                 use_bias=True,
                 name='{}/c4_lat'.format(name))

    x2 = nn.Lambda(lambda img: tf.image.resize_bilinear(img, [20, 20],
                                                        align_corners=True,
                                                        name='{}/c5_resize'.format(name)))(x2)
    x2 = conv1x1(x2,
                 in_channels=x2.shape[3],
                 out_channels=245,
                 strides=1,
                 groups=1,
                 use_bias=True,
                 name='{}/c5_lat'.format(name))

    zero = K.zeros((1, size, size, 528))
    x3 = nn.Lambda(lambda img: nn.add([img, zero]))(x3)
    x3 = conv1x1(x3,
                 in_channels=x3.shape[3],
                 out_channels=245,
                 strides=1,
                 groups=1,
                 use_bias=True,
                 name='{}/c_glb_lat'.format(name))
    print(x1)
    return nn.add([x1, x2, x3])


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 name="shuffle_unit"):
    mid_channels = out_channels // 2

    if downsample:
        y1 = depthwise_conv5x5(
            x=x,
            channels=in_channels,
            strides=2,
            name=name + "/dw_conv4")
        y1 = batchnorm(
            x=y1,
            name=name + "/dw_bn4")
        y1 = conv1x1(
            x=y1,
            in_channels=in_channels,
            out_channels=mid_channels,
            name=name + "/expand_conv5")
        y1 = batchnorm(
            x=y1,
            name=name + "/expand_bn5")
        y1 = nn.Activation("relu", name=name + "/expand_activ5")(y1)
        x2 = x
    else:
        in_split2_channels = in_channels // 2
        if is_channels_first():
            y1 = nn.Lambda(lambda z: z[:, 0:in_split2_channels, :, :])(x)
            x2 = nn.Lambda(lambda z: z[:, in_split2_channels:, :, :])(x)
        else:
            y1 = nn.Lambda(lambda z: z[:, :, :, 0:in_split2_channels])(x)
            x2 = nn.Lambda(lambda z: z[:, :, :, in_split2_channels:])(x)

    y2 = conv1x1(
        x=x2,
        in_channels=(in_channels if downsample else mid_channels),
        out_channels=mid_channels,
        name=name + "/compress_conv1")
    y2 = batchnorm(
        x=y2,
        name=name + "/compress_bn1")
    y2 = nn.Activation("relu", name=name + "/compress_activ1")(y2)

    y2 = depthwise_conv5x5(
        x=y2,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        name=name + "/dw_conv2")
    y2 = batchnorm(
        x=y2,
        name=name + "/dw_bn2")

    y2 = conv1x1(
        x=y2,
        in_channels=mid_channels,
        out_channels=mid_channels,
        name=name + "/expand_conv3")
    y2 = batchnorm(
        x=y2,
        name=name + "/expand_bn3")
    y2 = nn.Activation("relu", name=name + "/expand_activ3")(y2)

    if use_se:
        y2 = se_block(
            x=y2,
            channels=mid_channels,
            name=name + "/se")

    if use_residual and not downsample:
        y2 = nn.add([y2, x2], name=name + "/add")

    x = nn.concatenate([y1, y2], axis=get_channel_axis(), name=name + "/concat")

    x = channel_shuffle_lambda(
        channels=out_channels,
        groups=2,
        name=name + "/c_shuffle")(x)

    return x


def shuffle_init_block(x,
                       in_channels,
                       out_channels,
                       name="shuffle_init_block"):
    x = conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        name=name + "/conv")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=0,
        ceil_mode=True,
        name=name + "/pool")
    return x


def shufflenetv2(x,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 use_se=False,
                 use_residual=False,
                 in_channels=3,
                 in_size=(320, 320),
                 classes=2,
                 model_name="snet_146"):
    # input_shape = (in_channels, 320, 320) if is_channels_first() else (320, 320, in_channels)
    # input = nn.Input(shape=input_shape)

    x = shuffle_init_block(
        x,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels

    count_stage = 1
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            downsample = (j == 0)
            x = shuffle_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=downsample,
                use_se=use_se,
                use_residual=use_residual,
                name="features/stage{}/unit{}".format(i + 2, j + 1))
            print(x.shape)
            in_channels = out_channels
        count_stage += 1
        if count_stage == 3:
            c4 = x
        elif count_stage == 4:
            c5 = x

    if model_name == 'snet_49':
        x = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=final_block_channels,
            name="features/final_block")
        in_channels = final_block_channels
        # print(in_channels)
    else:
        in_channels = 1

    x = nn.GlobalAveragePooling2D(name="features/final_pool")(x)
    c_glb = x

    # x = flatten(x)
    # x = nn.Dense(
    #     units=classes,
    #     input_dim=in_channels,
    #     name="output")(x)

    y_cem = context_enhancement_module(x1=c4,
                                       x2=c5,
                                       x3=c_glb,
                                       size=20)
    return y_cem


def get_shufflenetv2(x,
                     width_scale,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join('~', '.keras', 'models'),
                     **kwargs):

    init_block_channels = 24
    final_block_channels = 512
    layers = [4, 8, 4]
    channels_per_layers = [132, 264, 528]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    # print(channels)
    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        if width_scale > 1.5:
            final_block_channels = int(final_block_channels * width_scale)
    # print(channels)
    net = shufflenetv2(
        x,
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        model_name=model_name,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def snet_49(**kwargs):
    return get_shufflenetv2(width_scale=(15.0 / 33.0), model_name="snet_49", **kwargs)


def snet_146(x, **kwargs):
    return get_shufflenetv2(x, width_scale=1.0, model_name="snet_146", **kwargs)


def snet_535(**kwargs):
    return get_shufflenetv2(width_scale=(62.0 / 33.0), model_name="snet_535", **kwargs)
