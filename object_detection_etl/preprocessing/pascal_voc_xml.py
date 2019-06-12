#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2019/5/5 18:42
@Author : zhanghongke
@Email : zhanghongke@dianwoda.com
@File : pascal_voc_xml.py
@Desc: 
'''

import cv2 as cv
from PIL import Image
from xml.etree.ElementTree import ElementTree,Element
import os
def read_xml(in_path):
    '''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree
def xml_show(ImgPath,AnnoPath):
    """
    the method to show the box from xml
    :return:
    """
    ImgPath ='/home/bigdata/zhk/tmp/labelimgdata/images/'
    AnnoPath ='/home/bigdata/zhk/tmp/labelimgdata/xmls/'
    pic_path=ImgPath+'1-457.png'
    xml_path=AnnoPath+'1-457.xml'
    img = Image.open(pic_path)
    imgSize = img.size
    pic_height = max(imgSize)
    pic_width = min(imgSize)
    print(pic_height,pic_width)
    # img = cv.imread(imgfile)
    # sp = img.shape
    # print sp
    # height = sp[0]#height(rows) of image
    # width = sp[1]#width(colums) of image
    # depth = sp[2]#the pixels value is made up of three primary color
    # print(height,width)
    out_path=AnnoPath+'new.xml'
    from xml.etree.ElementTree import ElementTree,Element
    # tree = ElementTree().parse(xml_path)
    tree = read_xml(xml_path)
    node1=tree.find('./size/width')
    node1.text = str(pic_width)
    node2 = tree.find('./size/height')#
    node2.text = str(pic_height)#注意这里要转换成str类型
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def update_xml_hw(ImgPath,AnnoPath):
    """
    the mothed to update the height and width of the xml
    :param ImgPath:
    :param AnnoPath:
    :return:
    """
    ImgPath = '/home/bigdata/zhk/tmp/labelimgdata/images/'
    AnnoPath = '/home/bigdata/zhk/tmp/labelimgdata/xmls/'
    # ImgPath ='/data2/rider_box_samples/VOCdevkit2007_20181023/VOCdevkit2007/VOC2007/JPEGImages/'
    # AnnoPath ='/data2/rider_box_samples/VOCdevkit2007_20181023/VOCdevkit2007/VOC2007/Annotations/'
    imagelist = os.listdir(ImgPath)
    print('imagelist', imagelist)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        print('image', image)
        print(image_pre, ext)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        # 读取图片
        print("imgfile", imgfile)
        img = Image.open(imgfile)
        imgSize = img.size
        pic_height = max(imgSize)
        pic_width = min(imgSize)
        print(pic_height, pic_width)
        # img = cv.imread(imgfile)
        # sp = img.shape
        # print sp
        # height = sp[0]  # height(rows) of image
        # width = sp[1]  # width(colums) of image
        # depth = sp[2]  # the pixels value is made up of three primary color
        # tree = ElementTree().parse(xml_path)
        tree = read_xml(xmlfile)
        node1 = tree.find('./size/width')
        node1.text = str(pic_width)
        node2 = tree.find('./size/height')  #
        node2.text = str(pic_height)  # 注意这里要转换成str类型
        print("xmlfile", xmlfile)
        tree.write(xmlfile, encoding="utf-8", xml_declaration=True)