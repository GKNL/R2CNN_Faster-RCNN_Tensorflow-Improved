# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 16:12
# @Author  : Peng Miao
# @File    : HRSC2VOC.py
# @Intro   : 【适用于HRSC2016数据集标注文件】将HRSC2016标注文件转换为适合该模型的VOC格式标注文件，并且从5点标注转换为8点标注
#            'mbox_ang'标签代表的是旋转角度（与x轴负轴的夹角大小，范围在-pi/2 ~ pi/2之间）


import os
import xml.etree.ElementTree as ET
import math
import numpy as np
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

import glob

# XML_ROOT_FOLDER = r'D:\ideaWorkPlace\Pycharm\graduation_project\R2CNN-DFPN_RPN_HEAD_AROI\data\testHRSC\Annotations'
XML_ROOT_FOLDER = r'G:\graduation_project_database\HRSC2016\HRSC2016\Test\Annotations'
XML_DES_FOLDER = r'D:\ideaWorkPlace\Pycharm\graduation_project\R2CNN_Faster-RCNN_Tensorflow-Improved\data\VOCdevkit\VOCdevkit_test\Annotations'
# SRC_IMAGE_ROOT_FOLDER = r"D:\ideaWorkPlace\Pycharm\graduation_project\R2CNN-DFPN_RPN_HEAD_AROI\data\testHRSC\images"
SRC_IMAGE_ROOT_FOLDER = r'G:\graduation_project_database\HRSC2016\HRSC2016\Test\AllImages'
DES_IMAGE_ROOT_FOLDER = r"D:\ideaWorkPlace\Pycharm\graduation_project\R2CNN_Faster-RCNN_Tensorflow-Improved\data\VOCdevkit\VOCdevkit_test\JPEGImages"
pi = 3.141592

# 解析文件名出来
def GetXMLFiles():
    global XML_ROOT_FOLDER
    file_list = []
    files = os.listdir(XML_ROOT_FOLDER)
    for file in files:
        if file.endswith(".xml"):
            # shutil.move(os.path.join(DATA_ROOT_FOLDER, file), os.path.join(DATA_ROOT_FOLDER, file.replace('.tif', '.tiff')))
            file_list.append(file)
    return file_list


def WriterXMLFiles(filename, Points, Truncateds, Headers, width, height):
    """
    将计算得到的坐标等信息，按照VOC格式写入xml文件
    :param filename: 对应的转换后的文件名
    :param Points: 转换后的坐标点集合(8坐标点表示法)
    :param Truncateds: '目标是否被截断'集合
    :param Headers: '舰船的头部坐标点'集合
    :param width: 标注文件对应的图片的宽
    :param height: 标注文件对应的图片的高
    :return:
    """

    # 创建Element，组装成新的xml文件
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')  # folder标签
    node_folder.text = 'train_images'
    node_filename = SubElement(node_root, 'filename')  # filename标签
    node_filename.text = filename.split(".")[0] + '.bmp'  # str(1) + '.jpg'

    # source标签
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = "xxxx"
    node_annotation = SubElement(node_source, 'annotation')
    node_annotation.text = "HRSC2016"
    node_image = SubElement(node_source, 'image')
    node_image.text = "HRSC2016"
    node_flickrid = SubElement(node_source, 'flickrid')
    node_flickrid.text = "0"

    # owner标签
    node_owner = SubElement(node_root, 'owner')
    node_flickrid2 = SubElement(node_owner, 'flickrid')
    node_flickrid2.text = "NEU_PM"
    node_name2 = SubElement(node_owner, 'name')
    node_name2.text = "NEU_PM"

    # size标签
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = "3"

    # segmented标签
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = "0"

    for (point, truncated, header) in zip(Points, Truncateds, Headers):
        # 开始构建标签，填入计算好的坐标

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'ship'
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unknown'
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = truncated
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')

        node_x1 = SubElement(node_bndbox, 'x0')
        node_x1.text = str(point[0])
        node_y1 = SubElement(node_bndbox, 'y0')
        node_y1.text = str(point[1])

        node_x2 = SubElement(node_bndbox, 'x1')
        node_x2.text = str(point[2])
        node_y2 = SubElement(node_bndbox, 'y1')
        node_y2.text = str(point[3])

        node_x3 = SubElement(node_bndbox, 'x2')
        node_x3.text = str(point[4])
        node_y3 = SubElement(node_bndbox, 'y2')
        node_y3.text = str(point[5])

        node_x4 = SubElement(node_bndbox, 'x3')
        node_x4.text = str(point[6])
        node_y4 = SubElement(node_bndbox, 'y3')
        node_y4.text = str(point[7])

        # node_xmin = SubElement(node_bndbox, 'xmin')
        # node_xmin.text = str(min(bowA_x, bowB_x, tailA_x, tailB_x))
        # node_ymin = SubElement(node_bndbox, 'ymin')
        # node_ymin.text = str(min(bowA_y, bowB_y, tailA_y, tailB_y))
        # node_xmax = SubElement(node_bndbox, 'xmax')
        # node_xmax.text = str(max(bowA_x, bowB_x, tailA_x, tailB_x))
        # node_ymax = SubElement(node_bndbox, 'ymax')
        # node_ymax.text = str(max(bowA_y, bowB_y, tailA_y, tailB_y))

        # 仅在Head分支时使用
        # node_headerX = SubElement(node_bndbox, 'head_x')
        # node_headerX.text = str(header[0])
        # node_headerY = SubElement(node_bndbox, 'head_y')
        # node_headerY.text = str(header[1])

    # break
    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)
    fw = open(os.path.join(XML_DES_FOLDER, filename), 'wb')
    fw.write(xml)
    fw.close()


def CropImage(image_path, des_path, Points, Truncateds, Headers, sub_image_h, sub_image_w, overlap_h, overlap_w):
    """
    裁剪图片
    :param image_path: 输入图片路径
    :param des_path: 输出图片路径
    :param Points: 单个xml中所有旋转标注框的四个顶点坐标集合
    :param Truncateds: 单个xml中所有旋转标注框中读出的Truncateds属性集合
    :param Headers: 单个xml中所有舰船目标的舰头坐标集合
    :param sub_image_h: 裁剪子图的高
    :param sub_image_w: 裁剪子图的宽
    :param overlap_h: 高度上可交叠的长度
    :param overlap_w: 宽度上可交叠的长度
    :return:
    """
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        print(image.shape)
        image_h, image_w, _ = image.shape  # 获取图片长宽
        for hh in range(0, image_h, sub_image_h - overlap_h):
            if (hh + sub_image_h) > image_h:
                break
            for ww in range(0, image_w, sub_image_w - overlap_w):
                if (ww + sub_image_w) > image_w:
                    break
                sub_image = image[hh:(hh + sub_image_h), ww:(ww + sub_image_w), :]
                sub_boxes = []
                sub_headers = []
                for inx, box in enumerate(Points):
                    box = np.array(box)
                    box = np.reshape(box, [-1, 2])
                    xmin = min(box[:-1, 0])
                    xmax = max(box[:-1, 0])
                    ymin = min(box[:-1, 1])
                    ymax = max(box[:-1, 1])
                    # if (int(IsPointInRect(box[0], [ww, hh, ww+sub_image_w, hh+sub_image_h])) +
                    #         int(IsPointInRect(box[1], [ww, hh, ww+sub_image_w, hh+sub_image_h])) +
                    #         int(IsPointInRect(box[2], [ww, hh, ww+sub_image_w, hh+sub_image_h])) +
                    #         int(IsPointInRect(box[3], [ww, hh, ww+sub_image_w, hh+sub_image_h]))) >= 2:
                    print("ww:"+str(ww)+",hh:"+str(hh)+",xmin:"+str(xmin)+",xmax:"+str(xmax)+",ymin:"+str(ymin)+",ymax:"+str(ymax))
                    if (ww < xmin) and (hh < ymin) and (ww + sub_image_w > xmax) and (hh + sub_image_h > ymax):
                        # 处理剪裁后的顶点坐标
                        box = np.array(box)
                        box = np.reshape(box, [-1, 2])  # 转换为n行2列的矩阵
                        box[:, 0] -= ww  # 横坐标裁剪去横向移动的步幅
                        box[:, 1] -= hh  # 纵坐标裁剪去纵向移动的步幅
                        box = np.reshape(box, [-1, ])  # 从n行2列矩阵转换回一行n列
                        sub_boxes.append(list(box))
                        # 处理剪裁后的舰头坐标
                        header_box = np.array(Headers)
                        header_box = np.reshape(header_box, [-1, 2])
                        header_box[:, 0] -= ww  # 横坐标裁剪去横向移动的步幅
                        header_box[:, 1] -= hh  # 纵坐标裁剪去纵向移动的步幅
                        header_box = np.reshape(header_box, [-1, ])  # 从n行2列矩阵转换回一行n列
                        sub_headers.append(list(header_box))

                # 过滤掉剪裁后被截断/没有目标对象的图片和xml文件，不写入和生成相应的文件
                if len(sub_boxes) != 0:
                    sub_image_name = image_path.split('\\')[-1].split('.')[0] + '%{}%{}'.format(ww, hh) + '.jpg'
                    sub_xml_name = image_path.split('\\')[-1].split('.')[0] + '%{}%{}'.format(ww, hh) + '.xml'
                    cv2.imwrite(des_path + "\\" + sub_image_name, sub_image)
                    print(sub_xml_name)
                    WriterXMLFiles(sub_xml_name, sub_boxes, Truncateds, sub_headers, sub_image_w, sub_image_h)


if __name__ == "__main__":

    # origin_ann_dir = 'voc_ship/JPEGImages/'
    # new_ann_dir = 'text/'

    xml_Lists = GetXMLFiles()
    print("总的xml文件数量： ", len(xml_Lists))

    xml_basenames = []  # e.g. 100.xml
    for item in xml_Lists:
        xml_basenames.append(os.path.basename(item))

    xml_names = []  # e.g. 100
    for item in xml_basenames:
        temp1, temp2 = os.path.splitext(item)
        xml_names.append(temp1)

    count = 0

    for it in xml_names:
        print("---------------------------{}图片及标注文件处理开始-------------------------------".format(it))
        tree = ET.parse(os.path.join(XML_ROOT_FOLDER, str(it) + '.xml'))
        root = tree.getroot()

        Points = []  # 转换好的坐标点集合
        Truncateds = []  # '目标是否被截断'集合
        Headers = []  # 舰船头部中点坐标集合
        # 对于每一个xml文件
        for Object in root.findall('./HRSC_Objects/HRSC_Object'):
            mbox_cx = float(Object.find('mbox_cx').text)
            mbox_cy = float(Object.find('mbox_cy').text)
            mbox_w = float(Object.find('mbox_w').text)
            mbox_h = float(Object.find('mbox_h').text)
            mbox_ang = float(Object.find('mbox_ang').text)
            # print("原始坐标：[{},{},{},{},{}]".format(mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang))

            # 计算舰首与舰尾点的坐标

            bow_x = mbox_cx + mbox_w / 2 * math.cos(mbox_ang)
            bow_y = mbox_cy + mbox_w / 2 * math.sin(mbox_ang)

            tail_x = mbox_cx - mbox_w / 2 * math.cos(mbox_ang)
            tail_y = mbox_cy - mbox_w / 2 * math.sin(mbox_ang)

            # print('舰首舰尾坐标：[{},{},{},{}]'.format(bow_x, bow_y, tail_x, tail_y))

            # 根据舰首舰尾的坐标，结合宽高，计算旋转矩形框四个顶点的坐标

            bowA_x = round(bow_x + mbox_h / 2 * math.sin(mbox_ang))
            bowA_y = round(bow_y - mbox_h / 2 * math.cos(mbox_ang))

            bowB_x = round(bow_x - mbox_h / 2 * math.sin(mbox_ang))
            bowB_y = round(bow_y + mbox_h / 2 * math.cos(mbox_ang))

            tailA_x = round(tail_x + mbox_h / 2 * math.sin(mbox_ang))
            tailA_y = round(tail_y - mbox_h / 2 * math.cos(mbox_ang))

            tailB_x = round(tail_x - mbox_h / 2 * math.sin(mbox_ang))
            tailB_y = round(tail_y + mbox_h / 2 * math.cos(mbox_ang))

            # print("转换后的坐标：[{},{},{},{},{},{},{},{}]".format(bowA_x, bowA_y, bowB_x, bowB_y, tailA_x, tailA_y, tailB_x, tailB_y))

            Points.append([bowA_x, bowA_y, bowB_x, bowB_y, tailA_x, tailA_y, tailB_x, tailB_y])
            Truncateds.append(Object.find("truncated").text)
            Headers.append([int(Object.find('header_x').text), int(Object.find('header_y').text)])

        # 剪裁图片，生成xml
        image_path = SRC_IMAGE_ROOT_FOLDER + r'\{}.bmp'.format(it)
        sub_image_h = 800
        sub_image_w = 800
        overlap_h = 600  # overlap越大，裁剪步幅越小
        overlap_w = 600

        CropImage(image_path, DES_IMAGE_ROOT_FOLDER, Points, Truncateds, Headers,
                  sub_image_h, sub_image_w, overlap_h, overlap_w)

        print("---------------------------{}文件处理结束-------------------------------".format(it))