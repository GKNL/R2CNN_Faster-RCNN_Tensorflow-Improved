# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import tfplot as tfp


def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def resnet_base(img_batch, scope_name, is_training=True):
    '''

    只输出了C4层作为Feature Map供后续RPN使用

    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              # use stride 1 for the last conv4 layer.

              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=1)]
              # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, _ = resnet_v1.resnet_v1(net,
                                    blocks[0:1],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, _ = resnet_v1.resnet_v1(C2,
                                    blocks[1:2],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(C3,
                                    blocks[2:3],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    return C4


def restnet_head(input, is_training, scope_name):
    block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, _ = resnet_v1.resnet_v1(input,
                                    block4,
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
        # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
        C5_flatten = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        # C5_flatten = tf.Print(C5_flatten, [tf.shape(C5_flatten)], summarize=10, message='C5_flatten_shape')

    # global average pooling C5 to obtain fc layers
    return C5_flatten


def resnet_dict(img_batch, scope_name, is_training=True):
    """
    类似于resnet_base，但是最后返回的是所有层feature组成的dict（resnet_base方法仅返回了C4层）
    :param img_batch:
    :param scope_name:
    :param is_training:
    :return:
    """
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. ')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4 - cfgs.FIXED_BLOCKS) * [True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    # self.add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    # self.add_heatmap(C3, name='Layer3/C3_heat')
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # self.add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
    # self.add_heatmap(C5, name='Layer5/C5_heat')

    feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_C4['{}/block3/unit_{}/bottleneck_v1'.format(scope_name, middle_num_units - 1)],
                    'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)],
                    # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                    }

    return feature_dict




# ***********************************************************************************************
# *                                  Standard FPN (By PM)                                       *
# ***********************************************************************************************

def add_heatmap(feature_maps, name):
    '''

    :param feature_maps:[B, H, W, C]
    :return:
    '''

    def figure_attention(activation):
        fig, ax = tfp.subplots()
        im = ax.imshow(activation, cmap='jet')
        fig.colorbar(im)
        return fig

    heatmap = tf.reduce_sum(feature_maps, axis=-1)
    heatmap = tf.squeeze(heatmap, axis=0)
    tfp.summary.plot(name, figure_attention, [heatmap])


def fusion_two_layer(C_i, P_j, scope):
    '''
    对Pj和Ci两个特征图进行拼接（这里是直接add相加）
    注：i = j+1
    :param C_i: shape is [1, h, w, c]
    :param P_j: shape is [1, h/2, w/2, 256]
    :return:
    P_i
    '''
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]
        h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
        upsample_p = tf.image.resize_bilinear(P_j,  # 以layer = 4为例: 对P5进行上采样使之与C4的size相同
                                              size=[h, w],
                                              name='up_sample_'+level_name)

        reduce_dim_c = slim.conv2d(C_i,  # 以layer = 4为例: 对C4进行1*1卷积，改变通道数
                                   num_outputs=256,
                                   kernel_size=[1, 1], stride=1,
                                   scope='reduce_dim_'+level_name)

        add_f = 0.5*upsample_p + 0.5*reduce_dim_c  # 直接相加（按照原文）（在DFPN中则使用concat）

        # P_i = slim.conv2d(add_f,
        #                   num_outputs=256, kernel_size=[3, 3], stride=1,
        #                   padding='SAME',
        #                   scope='fusion_'+level_name)
        return add_f


def resnet_feature_pyramid(img_batch, scope_name, is_training=True):
    """
    普通FPN实现
    this code is derived from FPN_Tensorflow.
    https://github.com/DetectionTeamUCAS/FPN_Tensorflow

    :param img_batch:
    :param scope_name:
    :param is_training:
    :return: pyramid feature list : [P2, P3, P4, P5, P6]
    """
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    add_heatmap(C3, name='Layer3/C3_heat')
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
    add_heatmap(C5, name='Layer5/C5_heat')

    feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_C4['{}/block3/unit_{}/bottleneck_v1'.format(scope_name, middle_num_units - 1)],
                    'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)],
                    # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                    }

    # -----------------------------------计算P2、P3、P4、P5、P6-----------------------------------
    pyramid_dict = {}
    with tf.variable_scope('build_standard_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            P5 = slim.conv2d(C5,
                             num_outputs=256,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')
            pyramid_dict['P5'] = P5

            if "P6" in cfgs.LEVELS:
                P6 = slim.max_pool2d(P5, kernel_size=[1, 1], stride=2, scope='build_P6')
                pyramid_dict['P6'] = P6

            for level in range(4, 1, -1):  # build [M4, M3, M2]

                pyramid_dict['P%d' % level] = fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                               P_j=pyramid_dict["P%d" % (level + 1)],
                                                               scope='build_P%d' % level)
            for level in range(5, 1, -1):  # 对Mi特征图再经过3 x 3卷积(减轻最近邻近插值带来的混叠影响，周围的数都相同)，得到最终的Pi
                pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                          num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                          stride=1, scope="fuse_P%d" % level)
    for level in range(5, 1, -1):
        add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_heat' % (level, level))

    # return [P2, P3, P4, P5, P6]
    print("we are in Standard Pyramid::-======>>>>")
    print(cfgs.LEVELS)
    print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
    print(20 * "__")
    return [pyramid_dict[level_name] for level_name in cfgs.LEVELS]  # return list rather than dict, to avoid dict is unordered


# ***********************************************************************************************
# *                        Dense FPN (By PM, reference to YangXue)                              *
# ***********************************************************************************************
def resnet_dense_feature_pyramid(img_batch, scope_name, is_training=True):
    """
    DFPN实现代码
    Reference: https://github.com/yangxue0827/R2CNN_HEAD_FPN_Tensorflow
    :param img_batch:
    :param scope_name:
    :param is_training:
    :return: dense pyramid feature list : [P2, P3, P4, P5, P6]
    """
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    add_heatmap(C3, name='Layer3/C3_heat')
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
    add_heatmap(C5, name='Layer5/C5_heat')

    feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_C4['{}/block3/unit_{}/bottleneck_v1'.format(scope_name, middle_num_units - 1)],
                    'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)],
                    # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                    }

    # -----------------------------------计算P2、P3、P4、P5、P6-----------------------------------
    pyramid_dict = {}
    with tf.variable_scope('build_dense_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):
            # C5层先经过1 x 1卷积，将通道数改为256，得到P5
            P5 = slim.conv2d(C5,
                             num_outputs=256,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')
            pyramid_dict['P5'] = P5

            if "P6" in cfgs.LEVELS:
                # 对P5进行max pooling 得到P6
                P6 = slim.max_pool2d(P5, kernel_size=[1, 1], stride=2, scope='build_P6')
                pyramid_dict['P6'] = P6

            for layer in range(4, 1, -1):  # 依次对C4, C3, C2进行处理，得到P4, P3, P2
                c = feature_dict['C' + str(layer)]
                # 以layer = 3为例，对C3进行1*1卷积，改变通道数
                c_conv = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                     scope='build_P%d/reduce_dimension' % layer)
                p_concat = [c_conv]
                up_sample_shape = tf.shape(c)

                # 下面的代码是DFPN的创新点，区别于普通的FPN
                for layer_top in range(5, layer, -1):  # 对P5和P4分别进行上采样（以layer = 3为例时）
                    p_temp = pyramid_dict['P' + str(layer_top)]
                    # 对P_temp进行上采样（使用最邻近插值）也可使用tf.image.resize_bilinear
                    p_sub = tf.image.resize_nearest_neighbor(p_temp, [up_sample_shape[1], up_sample_shape[2]],
                                                             name='build_P%d/up_sample_nearest_neighbor' % layer)
                    p_concat.append(p_sub)  # 将待concat的feature map加入list中

                # 将P5_sub, P4_sub, C3_conv进行拼接（以layer = 3为例时）
                p = tf.concat(p_concat, axis=3)  # 这里，img的shape为(1, H, W, C)，即在channel的维度上进行拼接
                # 对拼接后的结果再进行3*3的卷积（减轻最近邻近插值带来的混叠影响，周围的数都相同）
                p_conv = slim.conv2d(p, 256, kernel_size=[3, 3], stride=[1, 1],  # 所有Feature Map(Pi)的channel都是256
                                     padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                pyramid_dict['P' + str(layer)] = p_conv

    for level in range(5, 1, -1):
        add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/Dense P%d_heat' % (level, level))

    # return [P2, P3, P4, P5, P6]
    print("we are in Dense Pyramid::-======>>>>")
    print(cfgs.LEVELS)
    print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
    print(20 * "__")
    return [pyramid_dict[level_name] for level_name in cfgs.LEVELS]  # return list rather than dict, to avoid dict is unordered


















