# -*- coding: utf-8 -*-
# @Time    : 2021/3/16 17:17
# @Author  : Peng Miao
# @File    : check_objects.py
# @Intro   :

import os
import xml.etree.ElementTree as ET
import math
import numpy as np
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

XML_ROOT_FOLDER = r'D:\ideaWorkPlace\Pycharm\graduation_project\R2CNN_Faster-RCNN_Tensorflow-Improved\data\VOCdevkit\VOCdevkit_train\Annotations'

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
    tree = ET.parse(os.path.join(XML_ROOT_FOLDER, str(it) + '.xml'))
    root = tree.getroot()

    # 对于每一个xml文件
    objects = root.findall('./object')
    print("{}图片含有船舶{}个".format(it, len(objects)))
    if(len(objects)==0):
        count += 1

print(count)