# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.networks.rpn import rpn_utils
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer import proposal_target_layer
from libs.box_utils import nms_rotate
from libs.models import neck_fpn, scrdet_neck


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):

            feature_dict = resnet.resnet_dict(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

            if cfgs.FPN_MODE == 'FPN':  # 使用普通版FPN提取Feature Map [返回结果为feature map list]
                fpn_func = neck_fpn.NeckFPN(cfgs)
                return fpn_func.fpn(feature_dict, self.is_training)

            elif cfgs.FPN_MODE == 'DFPN':  # 使用DFPN提取Feature Map [返回结果为feature map list]
                fpn_func = neck_fpn.NeckFPN(cfgs)
                return fpn_func.dense_fpn(feature_dict, self.is_training)

            elif cfgs.FPN_MODE == 'SCRDet':  # 使用SF-Net + MDA-Net提取Feature Map [返回结果为单张feature map]
                fpn_func = scrdet_neck.NeckSCRDet(cfgs)
                return fpn_func.scrdet_fpn(feature_dict, self.is_training)

            elif cfgs.FPN_MODE == 'Resnet_C4':  # 直接使用Resnet的C4作为Feature Map
                return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name,
                                                  is_training=self.is_training)
            else:
                raise Exception('only support [DFPN, FPN, SCRDet, Resnet C4]')

            # 直接使用C4作为Feature Map
            # return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

            # # 使用FPN提取出Feature Map list
            # return resnet.resnet_feature_pyramid(input_img_batch, scope_name=self.base_network_name,
            #                                      is_training=self.is_training)

            # # 使用FPN提取出Feature Map list
            # return resnet.resnet_dense_feature_pyramid(input_img_batch, scope_name=self.base_network_name,
            #                                            is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_fastrcnn_h(self, rois, bbox_ppred, scores, img_shape):
        '''
        FastRCNN后处理阶段（水平分支）
        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_h'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encode_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def postprocess_fastrcnn_r(self, rois, bbox_ppred, scores, img_shape):
        '''
        FastRCNN后处理阶段（旋转分支）
        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 5]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_r'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 5])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes_rotate(encode_boxes=tmp_encoded_box,
                                                                          reference_boxes=rois,
                                                                          scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                # tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                #                                                              img_shape=img_shape)

                # 3. NMS
                keep = nms_rotate.nms_rotate(decode_boxes=tmp_decoded_boxes,
                                             scores=tmp_score,
                                             iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             use_angle_condition=False,
                                             angle_threshold=15,
                                             use_gpu=cfgs.ROTATE_NMS_USE_GPU)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape, scope=""):
        '''
        这里用的是ROI Warping(介于ROI Pooling和ROI Align之间：RoIWarp是将RoI量化到feature map上)
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop  【A 4-D tensor of shape `[batch, channel, image_height, image_width]`】
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]  proposal坐标
        :return:
        '''

        with tf.variable_scope('ROI_Warping'+scope):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w  # 标准化坐标
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(  # 矩阵转置
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)  # 标准化后的proposal坐标

            # 卷积特征图相应部分被裁剪，并resize为常数大小（14,14）
            cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ],
                                                                             dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )  # method默认为'bilinear'
            # 用2×2的滑动窗口进行最大池化操作，输出的尺度是7×7
            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)

        return roi_features

    def build_fastrcnn(self, feature_to_cropped, rois, img_shape):

        with tf.variable_scope('Fast-RCNN'):
            # 5. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                if cfgs.FPN_MODE == "FPN" or cfgs.FPN_MODE == "DFPN":  # FPN情况下递归对每层Feature Map进行ROI Pooling
                    pooled_features_list = []
                    for level_name, p, rois in zip(cfgs.LEVELS, feature_to_cropped, rois):  # exclude P6_rois
                        # p = tf.Print(p, [tf.shape(p)], summarize=10, message=level_name+'SHPAE***')
                        pooled_features = self.roi_pooling(feature_maps=p, rois=rois, img_shape=img_shape,
                                                           scope=level_name)
                        pooled_features_list.append(pooled_features)

                    pooled_features = tf.concat(pooled_features_list, axis=0)  # [minibatch_size, H, W, C]
                else:
                    pooled_features = self.roi_pooling(feature_maps=feature_to_cropped, rois=rois, img_shape=img_shape)

            # 6. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):
                fc_flatten = resnet.restnet_head(input=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name)
            elif self.base_network_name.startswith('MobilenetV2'):
                fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                           is_training=self.is_training)
            else:
                raise NotImplementedError('only support resnet and mobilenet')

            # 7. cls and reg in Fast-RCNN
            with tf.variable_scope('horizen_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                    cls_score_h = slim.fully_connected(fc_flatten,
                                                       num_outputs=cfgs.CLASS_NUM+1,
                                                       weights_initializer=cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='cls_fc_h')

                    bbox_pred_h = slim.fully_connected(fc_flatten,
                                                       num_outputs=(cfgs.CLASS_NUM+1) * 4,
                                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='reg_fc_h')
                    # for convient. It also produce (cls_num +1) bboxes

                    cls_score_h = tf.reshape(cls_score_h, [-1, cfgs.CLASS_NUM+1])
                    bbox_pred_h = tf.reshape(bbox_pred_h, [-1, 4*(cfgs.CLASS_NUM+1)])

            with tf.variable_scope('rotation_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                    cls_score_r = slim.fully_connected(fc_flatten,
                                                       num_outputs=cfgs.CLASS_NUM + 1,
                                                       weights_initializer=cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='cls_fc_r')

                    bbox_pred_r = slim.fully_connected(fc_flatten,
                                                       num_outputs=(cfgs.CLASS_NUM + 1) * 5,
                                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='reg_fc_r')
                    # for convient. It also produce (cls_num +1) bboxes
                    cls_score_r = tf.reshape(cls_score_r, [-1, cfgs.CLASS_NUM + 1])
                    bbox_pred_r = tf.reshape(bbox_pred_r, [-1, 5 * (cfgs.CLASS_NUM + 1)])

            return bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r

    def assign_levels(self, all_rois, labels=None, bbox_targets=None):
        '''
        对RPN生成的proposals（ROIs）进行一个分类，找出它们各自来自于FPN的哪一层
        :param all_rois:  RPN生成的所有Proposals
        :param labels:  Proposal对应的label
        :param bbox_targets:  Proposal对应的ground truth targets
        :return:
        '''
        with tf.name_scope('assign_levels'):
            # all_rois = tf.Print(all_rois, [tf.shape(all_rois)], summarize=10, message='ALL_ROIS_SHAPE*****')
            xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)

            h = tf.maximum(0., ymax - ymin)
            w = tf.maximum(0., xmax - xmin)

            levels = tf.floor(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))  # 4 + log_2(***)
            # use floor instead of round

            min_level = int(cfgs.LEVLES[0][-1])
            max_level = min(5, int(cfgs.LEVLES[-1][-1]))
            levels = tf.maximum(levels, tf.ones_like(levels) * min_level)  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * max_level)  # level maximum is 5

            levels = tf.stop_gradient(tf.reshape(levels, [-1]))

            def get_rois(levels, level_i, rois, labels, bbox_targets):

                level_i_indices = tf.reshape(tf.where(tf.equal(levels, level_i)), [-1])
                # level_i_indices = tf.Print(level_i_indices, [tf.shape(tf.where(tf.equal(levels, level_i)))[0]], message="SHAPE%d***"%level_i,
                #                            summarize=10)
                tf.summary.scalar('LEVEL/LEVEL_%d_rois_NUM' % level_i, tf.shape(level_i_indices)[0])
                level_i_rois = tf.gather(rois, level_i_indices)

                if self.is_training:
                    if cfgs.CUDA9:
                        # Note: for cuda 9
                        level_i_rois = tf.stop_gradient(level_i_rois)
                        level_i_labels = tf.gather(labels, level_i_indices)

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                    else:

                        # Note: for cuda 8
                        level_i_rois = tf.stop_gradient(tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0))
                        # to avoid the num of level i rois is 0.0, which will broken the BP in tf

                        level_i_labels = tf.gather(labels, level_i_indices)
                        level_i_labels = tf.stop_gradient(tf.concat([level_i_labels, [0]], axis=0))

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                        level_i_targets = tf.stop_gradient(tf.concat([level_i_targets,
                                                                      tf.zeros(shape=(1, 4 * (cfgs.CLASS_NUM + 1)),
                                                                               dtype=tf.float32)], axis=0))

                    return level_i_rois, level_i_labels, level_i_targets
                else:
                    if not cfgs.CUDA9:
                        # Note: for cuda 8
                        level_i_rois = tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0)
                    return level_i_rois, None, None

            rois_list = []
            labels_list = []
            targets_list = []
            for i in range(min_level, max_level + 1):
                P_i_rois, P_i_labels, P_i_targets = get_rois(levels, level_i=i, rois=all_rois,
                                                             labels=labels,
                                                             bbox_targets=bbox_targets)
                rois_list.append(P_i_rois)
                labels_list.append(P_i_labels)
                targets_list.append(P_i_targets)

            if self.is_training:
                all_labels = tf.concat(labels_list, axis=0)
                all_targets = tf.concat(targets_list, axis=0)
                return rois_list, all_labels, all_targets
            else:
                return rois_list  # [P2_rois, P3_rois, P4_rois, P5_rois] Note: P6 do not assign rois

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.draw_box_with_color(img, positive_anchor, tf.shape(positive_anchor)[0])
        neg_in_img = show_box_in_tensor.draw_box_with_color(img, negative_anchor, tf.shape(positive_anchor)[0])

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = show_box_in_tensor.draw_box_with_color(img, pos_roi, tf.shape(pos_roi)[0])
        neg_in_img = show_box_in_tensor.draw_box_with_color(img, neg_roi, tf.shape(neg_roi)[0])

        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def make_anchors(self, feature_maps):
        if cfgs.FPN_MODE == "FPN" or cfgs.FPN_MODE == "DFPN":  # Feature Pyramid，需要迭代各层以生成anchors
            all_anchors = []
            for i in range(len(cfgs.LEVELS)):
                level_name, p = cfgs.LEVELS[i], feature_maps[i]

                p_h, p_w = tf.shape(p)[1], tf.shape(p)[2]
                featuremap_height = tf.cast(p_h, tf.float32)
                featuremap_width = tf.cast(p_w, tf.float32)
                anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[i],
                                                    anchor_scales=cfgs.ANCHOR_SCALES,
                                                    anchor_ratios=cfgs.ANCHOR_RATIOS,
                                                    featuremap_height=featuremap_height,
                                                    featuremap_width=featuremap_width,
                                                    stride=cfgs.ANCHOR_STRIDE_LIST[i],
                                                    name="make_anchors_for%s" % level_name)
                all_anchors.append(anchors)
            all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors_of_FPN')
            return all_anchors
        else:  # 单层Feature Map，无需迭代
            featuremap_height, featuremap_width = tf.shape(feature_maps)[1], tf.shape(feature_maps)[2]
            featuremap_height = tf.cast(featuremap_height, tf.float32)
            featuremap_width = tf.cast(featuremap_width, tf.float32)

            anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                                anchor_scales=cfgs.ANCHOR_SCALES, anchor_ratios=cfgs.ANCHOR_RATIOS,
                                                featuremap_height=featuremap_height,
                                                featuremap_width=featuremap_width,
                                                stride=cfgs.ANCHOR_STRIDE,
                                                name="make_anchors_forRPN")
            return anchors

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred_h, bbox_targets_h, cls_score_h, bbox_pred_r, bbox_targets_r, cls_score_r, labels):
        '''
        计算loss值

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred_h: [-1, 4*(cls_num+1)]
        :param bbox_targets_h: [-1, 4*(cls_num+1)]
        :param cls_score_h: [-1, cls_num+1]
        :param bbox_pred_r: [-1, 5*(cls_num+1)]
        :param bbox_targets_r: [-1, 5*(cls_num+1)]
        :param cls_score_r: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                          bbox_targets=rpn_bbox_targets,
                                                          label=rpn_labels,
                                                          sigma=cfgs.RPN_SIGMA)
                # rpn_cls_loss:
                # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                # rpn_labels = tf.reshape(rpn_labels, [-1])
                # ensure rpn_labels shape is [-1]
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))

                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss_h = losses.smooth_l1_loss_rcnn_h(bbox_pred=bbox_pred_h,
                                                               bbox_targets=bbox_targets_h,
                                                               label=labels,
                                                               num_classes=cfgs.CLASS_NUM + 1,
                                                               sigma=cfgs.FASTRCNN_SIGMA)

                    # cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                    # labels = tf.reshape(labels, [-1])
                    cls_loss_h = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score_h,
                        labels=labels))  # beacause already sample before

                    bbox_loss_r = losses.smooth_l1_loss_rcnn_r(bbox_pred=bbox_pred_r,
                                                               bbox_targets=bbox_targets_r,
                                                               label=labels,
                                                               num_classes=cfgs.CLASS_NUM + 1,
                                                               sigma=cfgs.FASTRCNN_SIGMA)

                    cls_loss_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score_r,
                        labels=labels))
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss = bbox_loss = losses.sum_ohem_loss(
                        cls_score=cls_score_h,
                        label=labels,
                        bbox_targets=bbox_targets_h,
                        nr_ohem_sampling=128,
                        nr_classes=cfgs.CLASS_NUM + 1)

                cls_loss_h = cls_loss_h * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss_h = bbox_loss_h * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
                cls_loss_r = cls_loss_r * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss_r = bbox_loss_r * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss_h': cls_loss_h,
                'fastrcnn_loc_loss_h': bbox_loss_h,
                'fastrcnn_cls_loss_r': cls_loss_r,
                'fastrcnn_loc_loss_r': bbox_loss_r,
            }
        return loss_dict

    def build_whole_detection_network(self, input_img_batch, gtboxes_r_batch, gtboxes_h_batch, mask_batch=None):

        if self.is_training:
            # ensure shape is [M, 5] and [M, 6]
            gtboxes_r_batch = tf.reshape(gtboxes_r_batch, [-1, 6])
            gtboxes_h_batch = tf.reshape(gtboxes_h_batch, [-1, 5])
            gtboxes_r_batch = tf.cast(gtboxes_r_batch, tf.float32)
            gtboxes_h_batch = tf.cast(gtboxes_h_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network  提取特征图  (pa_mask仅在cfgs.FPN_MODE="SCRDet"的情况下返回)
        feature_maps, pa_mask = self.build_base_network(input_img_batch)  # single FP without FPN; FP list with FPN/DFPN

        # 2. build rpn
        # Done: 现在是针对于单张Feature Map进行操作计算RPN两个分支 -> 需要对每个level的Feature Map都进行操作

        # with tf.variable_scope('build_rpn',
        #                        regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
        #
        #     rpn_conv3x3 = slim.conv2d(
        #         feature_to_cropped, 512, [3, 3],
        #         trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
        #         activation_fn=tf.nn.relu,
        #         scope='rpn_conv/3x3')
        #     rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*2, [1, 1], stride=1,
        #                                 trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
        #                                 activation_fn=None,
        #                                 scope='rpn_cls_score')
        #     rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*4, [1, 1], stride=1,
        #                                trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
        #                                activation_fn=None,
        #                                scope='rpn_bbox_pred')
        #     rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
        #     rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        #     rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

        # 计算RPN两个分支的输出值（分类分支和回归分支）
        if cfgs.FPN_MODE == "FPN" or cfgs.FPN_MODE == "DFPN":
            rpn_box_pred, rpn_cls_score, rpn_cls_prob = rpn_utils.build_rpn_with_feature_pyramid(cfgs,
                                                                                                 feature_maps,
                                                                                                 self.is_training,
                                                                                                 self.num_anchors_per_location)
        else:
            rpn_box_pred, rpn_cls_score, rpn_cls_prob = rpn_utils.build_rpn_with_single_feature_map(cfgs,
                                                                                                    feature_maps,
                                                                                                    self.is_training,
                                                                                                    self.num_anchors_per_location)

        # 3. generate_anchors
        # Done: 现在是针对于单张Feature Map进行操作计算RPN生成的Anchors -> 需要对每个level的Feature Map都进行操作
        anchors = self.make_anchors(feature_maps)

        # 4. postprocess rpn proposals. such as: decode, clip, NMS  【RPN后处理阶段】
        with tf.variable_scope('postprocess_RPN'):

            # 最终RPN生成的proposals及其对应的分数
            rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                         rpn_cls_prob=rpn_cls_prob,
                                                         img_shape=img_shape,
                                                         anchors=anchors,
                                                         is_training=self.is_training)
            # rois shape [-1, 4]
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.is_training:
                rois_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=input_img_batch,
                                                                            boxes=rois,
                                                                            scores=roi_scores)
                tf.summary.image('all_rpn_rois', rois_in_img)

                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=input_img_batch,
                                                                                    boxes=score_gre_05_rois,
                                                                                    scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        ######################################################################################################
        #                                         sample minibatch
        # (分别sample出两批minibatch：一批用于RPN回归[只需要水平gt样本]，一批用于Fast RCNN回归[需要水平和旋转gt样本])
        ######################################################################################################

        if self.is_training:  # sample minibatch  读取用于RPN回归的小批量anchor样本。（后续根据gt和RPN输出的anchors计算平移和缩放因子）
            with tf.variable_scope('sample_anchors_minibatch'):
                # ground truth的坐标和对应标签
                rpn_labels, rpn_bbox_targets = \
                    tf.py_func(
                        anchor_target_layer,
                        [gtboxes_h_batch, img_shape, anchors],
                        [tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])  # gt target object的坐标：[x*,y*,w*,h*]
                rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
                rpn_labels = tf.reshape(rpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, rpn_labels)

            # --------------------------------------add smry-----------------------------------------------------------

            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
            rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/rpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):
                with tf.variable_scope('sample_RCNN_minibatch'):  # 读取用于Fast RCNN回归的小批量样本
                    # (bbox_targets_h是水平回归分支，可以去掉这个分支)
                    # 对这些sample出来的minibatch，找到它们对应的gt target
                    rois, labels, bbox_targets_h, bbox_targets_r = \
                    tf.py_func(proposal_target_layer,
                               [rois, gtboxes_h_batch, gtboxes_r_batch],
                               [tf.float32, tf.float32, tf.float32, tf.float32])

                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets_h = tf.reshape(bbox_targets_h, [-1, 4*(cfgs.CLASS_NUM+1)])
                    bbox_targets_r = tf.reshape(bbox_targets_r, [-1, 5*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels)


        # assign level   对RPN生成的proposals（ROIs）进行一个分类，找出它们各自来自于FPN的哪一层
        if cfgs.FPN_MODE == "FPN" or cfgs.FPN_MODE == "DFPN":
            if self.is_training:
                rois, labels, bbox_targets = self.assign_levels(all_rois=rois,
                                                                     labels=labels,
                                                                     bbox_targets=bbox_targets_r)
            else:
                rois = self.assign_levels(all_rois=rois)  # rois_list: [P2_rois, P3_rois, P4_rois, P5_rois]

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # 5. build Fast-RCNN
        # TODO:change to adapt to FPN.(其实就是ROI Pooling处需要加个循环)
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r = self.build_fastrcnn(feature_to_cropped=feature_maps,  # FPN情况下是P_List
                                                                                 rois=rois,  # RPN生成的proposals(FPN情况下是rois_list)
                                                                                 img_shape=img_shape)
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        # 对两个分类分支使用Softmax进行分类
        cls_prob_h = slim.softmax(cls_score_h, 'cls_prob_h')
        cls_prob_r = slim.softmax(cls_score_r, 'cls_prob_r')

        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            cls_category_h = tf.argmax(cls_prob_h, axis=1)
            fast_acc_h = tf.reduce_mean(tf.to_float(tf.equal(cls_category_h, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc_h', fast_acc_h)

            cls_category_r = tf.argmax(cls_prob_r, axis=1)
            fast_acc_r = tf.reduce_mean(tf.to_float(tf.equal(cls_category_r, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc_r', fast_acc_r)

        #  6. postprocess_fastrcnn
        if not self.is_training:
            final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                          bbox_ppred=bbox_pred_h,
                                                                                          scores=cls_prob_h,
                                                                                          img_shape=img_shape)
            final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                          bbox_ppred=bbox_pred_r,
                                                                                          scores=cls_prob_r,
                                                                                          img_shape=img_shape)
            return final_boxes_h, final_scores_h, final_category_h, final_boxes_r, final_scores_r, final_category_r
        else:
            '''
            when trian. We need build Loss
            '''
            loss_dict = self.build_loss(rpn_box_pred=rpn_box_pred,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred_h=bbox_pred_h,
                                        bbox_targets_h=bbox_targets_h,
                                        cls_score_h=cls_score_h,
                                        bbox_pred_r=bbox_pred_r,
                                        bbox_targets_r=bbox_targets_r,
                                        cls_score_r=cls_score_r,
                                        labels=labels)

            # build Attention Loss(SCRDet)
            if cfgs.FPN_MODE=="SCRDet" and self.is_training:
                with tf.variable_scope('build_attention_loss',
                                       regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                    attention_loss_c4 = losses.build_attention_loss(mask_batch, pa_mask)
                    attention_loss = attention_loss_c4
                    loss_dict['attention_loss'] = attention_loss

            final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                          bbox_ppred=bbox_pred_h,
                                                                                          scores=cls_prob_h,
                                                                                          img_shape=img_shape)
            final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                          bbox_ppred=bbox_pred_r,
                                                                                          scores=cls_prob_r,
                                                                                          img_shape=img_shape)

            return final_boxes_h, final_scores_h, final_category_h, \
                   final_boxes_r, final_scores_r, final_category_r, loss_dict

    def get_restorer(self):
        # 检查最新的训练好的权重文件
        # checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))
        checkpoint_path = "../output/trained_weights/R2CNN_20210316_HRSC2016_v1/voc_63000model.ckpt"  # 测试使用

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # print(model_variables)

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name+'/block4'):
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"---")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weights in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















