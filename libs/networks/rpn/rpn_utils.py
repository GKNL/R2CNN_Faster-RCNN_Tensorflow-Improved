# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 16:50
# @Author  : Peng Miao
# @File    : rpn_utils.py
# @Intro   : 用于build_whole_network.py中build_whole_detection_network的第二步和第三步：
#            Step 2 - build rpn: 根据单张feature map来计算rpn两个分支的输出值  or  对FPN的5层feature map分别计算两个分支的输出值
#            Step 3 - make Anchors: 使用FPN情况下生成anchor  or  不使用FPN情况下生成anchor

import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_rpn_with_single_feature_map(cfgs, feature_to_cropped, is_training, num_anchors_per_location):
    """
    对整个Feature Map进行卷积操作（分类和回归分支）[注意是对整个Feature Map进行操作，而不是对Anchor进行操作]
    因为Feature Map上每个点对应原图一个区域(k个anchor)，相当于对原图上的每个anchor进行分类和回归分支了
    :param cfgs:
    :param feature_to_cropped:
    :param is_training:
    :param num_anchors_per_location:
    :return:
    """
    rpn_conv3x3 = slim.conv2d(
        feature_to_cropped, 512, [3, 3],
        trainable=is_training, weights_initializer=cfgs.INITIALIZER,
        activation_fn=tf.nn.relu,
        scope='rpn_conv/3x3')
    # 分类分支：对rpn_conv2d_3x3进行分类（前景/非前景分数值）
    rpn_cls_score = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 2, [1, 1], stride=1,
                                trainable=is_training, weights_initializer=cfgs.INITIALIZER,
                                activation_fn=None,
                                scope='rpn_cls_score')
    # 回归分支（移变换t_x*,t_y*和缩放尺度t_w*,t_h*）
    rpn_box_pred = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 4, [1, 1], stride=1,
                               trainable=is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                               activation_fn=None,
                               scope='rpn_bbox_pred')
    rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
    rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
    rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

    return rpn_box_pred, rpn_cls_score, rpn_cls_prob


def build_rpn_with_feature_pyramid(cfgs, fp_list, is_training, num_anchors_per_location):
    with tf.variable_scope('build_rpn',
                           regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

        fpn_cls_score = []
        fpn_box_pred = []
        for level_name, p in zip(cfgs.LEVLES, fp_list):
            if cfgs.SHARE_HEADS:
                reuse_flag = None if level_name == cfgs.LEVLES[0] else True
                scope_list = ['rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred']
            else:
                reuse_flag = None
                scope_list = ['rpn_conv/3x3_%s' % level_name, 'rpn_cls_score_%s' % level_name,
                              'rpn_bbox_pred_%s' % level_name]
            rpn_conv3x3 = slim.conv2d(
                p, 512, [3, 3],
                trainable=is_training, weights_initializer=cfgs.INITIALIZER, padding="SAME",
                activation_fn=tf.nn.relu,
                scope=scope_list[0],
                reuse=reuse_flag)
            rpn_cls_score = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 2, [1, 1], stride=1,
                                        trainable=is_training, weights_initializer=cfgs.INITIALIZER,
                                        activation_fn=None, padding="VALID",
                                        scope=scope_list[1],
                                        reuse=reuse_flag)
            rpn_box_pred = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 4, [1, 1], stride=1,
                                       trainable=is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                       activation_fn=None, padding="VALID",
                                       scope=scope_list[2],
                                       reuse=reuse_flag)
            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

            fpn_cls_score.append(rpn_cls_score)
            fpn_box_pred.append(rpn_box_pred)

        fpn_cls_score = tf.concat(fpn_cls_score, axis=0, name='fpn_cls_score')
        fpn_box_pred = tf.concat(fpn_box_pred, axis=0, name='fpn_box_pred')
        fpn_cls_prob = slim.softmax(fpn_cls_score, scope='fpn_cls_prob')

        return fpn_box_pred, fpn_cls_score, fpn_cls_prob