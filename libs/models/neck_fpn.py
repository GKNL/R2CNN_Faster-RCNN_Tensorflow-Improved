# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 18:28
# @Author  : Peng Miao
# @File    : neck_fpn.py
# @Intro   :

from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf


class NeckFPN(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def fusion_two_layer(self, C_i, P_j, scope, is_training):
        '''
        i = j+1
        :param C_i: shape is [1, h, w, c]
        :param P_j: shape is [1, h/2, w/2, 256]
        :return:
        P_i
        '''
        with tf.variable_scope(scope):
            level_name = scope.split('_')[1]

            h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
            upsample_p = tf.image.resize_bilinear(P_j,
                                                  size=[h, w],
                                                  name='up_sample_'+level_name)

            reduce_dim_c = slim.conv2d(C_i,
                                       num_outputs=256,
                                       kernel_size=[1, 1], stride=1,
                                       trainable=is_training,
                                       scope='reduce_dim_'+level_name)

            add_f = 0.5*upsample_p + 0.5*reduce_dim_c

            return add_f

    def fpn(self, feature_dict, is_training):
        """
        普通FPN实现
        this code is derived from FPN_Tensorflow.
        https://github.com/DetectionTeamUCAS/FPN_Tensorflow
        :param feature_dict: Resnet提取出的特征图dict: [C2, C3...]
        :param is_training:
        :return:
        """

        pyramid_dict = {}
        with tf.variable_scope('build_standard_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY),
                                activation_fn=None, normalizer_fn=None):

                P5 = slim.conv2d(feature_dict['C5'],
                                 num_outputs=256,
                                 kernel_size=[1, 1],
                                 stride=1,
                                 trainable=is_training,
                                 scope='build_P5')

                if self.cfgs.ADD_GLOBAL_CTX:
                    print(10 * "ADD GLOBAL CTX.....")
                    global_ctx = tf.reduce_mean(feature_dict['C5'], axis=[1, 2], keep_dims=True)
                    global_ctx = slim.conv2d(global_ctx, kernel_size=[1, 1], num_outputs=256, stride=1,
                                             activation_fn=None, trainable=is_training, scope='global_ctx')
                    pyramid_dict['P5'] = P5 + global_ctx
                else:
                    pyramid_dict['P5'] = P5

                for level in range(4, int(self.cfgs.LEVELS[0][-1]) - 1, -1):  # build [P4, P3, P2]

                    pyramid_dict['P%d' % level] = self.fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                                        P_j=pyramid_dict["P%d" % (level + 1)],
                                                                        scope='build_P%d' % level,
                                                                        is_training=is_training)

                # 对Pi特征图再经过3 x 3卷积(减轻最近邻近插值带来的混叠影响，周围的数都相同)，得到最终的Pi
                for level in range(5, int(self.cfgs.LEVELS[0][-1]) - 1, -1):  # use 3x3 conv fuse P5, P4, P3, P2
                    pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                              num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                              stride=1, trainable=is_training,
                                                              scope="fuse_P%d" % level)
                if "P6" in self.cfgs.LEVELS:
                    # if use supervised_mask, we get p6 after enlarge RF
                    pyramid_dict['P6'] = slim.avg_pool2d(pyramid_dict["P5"], kernel_size=[2, 2],
                                                         stride=2, scope='build_P6')
        # for level in range(5, 1, -1):
        #     add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_fpn_heat' % (level, level))

        # return [P2, P3, P4, P5, P6]
        print("we are in Standard Pyramid::-======>>>>")
        print(self.cfgs.LEVELS)
        print("base_anchor_size are: ", self.cfgs.BASE_ANCHOR_SIZE_LIST)
        print(20 * "__")

        return [pyramid_dict[level_name] for level_name in self.cfgs.LEVELS]  # return list rather than dict, to avoid dict is unordered

    def dense_fpn(self, feature_dict, is_training):
        """
        DFPN实现代码
        Reference: https://github.com/yangxue0827/R2CNN_HEAD_FPN_Tensorflow
        :param feature_dict:
        :param is_training:
        :return:
        """
        pyramid_dict = {}
        with tf.variable_scope('build_dense_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY),
                                activation_fn=None, normalizer_fn=None):

                P5 = slim.conv2d(feature_dict['C5'],
                                 num_outputs=256,
                                 kernel_size=[1, 1],
                                 stride=1,
                                 trainable=is_training,
                                 scope='build_P5')

                if self.cfgs.ADD_GLOBAL_CTX:
                    print(10 * "ADD GLOBAL CTX.....")
                    global_ctx = tf.reduce_mean(feature_dict['C5'], axis=[1, 2], keep_dims=True)
                    global_ctx = slim.conv2d(global_ctx, kernel_size=[1, 1], num_outputs=256, stride=1,
                                             activation_fn=None, trainable=is_training, scope='global_ctx')
                    pyramid_dict['P5'] = P5 + global_ctx
                else:
                    pyramid_dict['P5'] = P5

                if "P6" in self.cfgs.LEVELS:
                    # if use supervised_mask, we get p6 after enlarge RF
                    pyramid_dict['P6'] = slim.avg_pool2d(pyramid_dict["P5"], kernel_size=[2, 2],
                                                         stride=2, scope='build_P6')

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

        # for level in range(5, 1, -1):
        #     add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_fpn_heat' % (level, level))

        # return [P2, P3, P4, P5, P6]
        print("we are in Dense Pyramid::-======>>>>")
        print(self.cfgs.LEVELS)
        print("base_anchor_size are: ", self.cfgs.BASE_ANCHOR_SIZE_LIST)
        print(20 * "__")

        return [pyramid_dict[level_name] for level_name in self.cfgs.LEVELS]  # return list rather than dict, to avoid dict is unordered