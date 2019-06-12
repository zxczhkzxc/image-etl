#coding:utf-8

from xml.dom.minidom import Document
import cv2
import os
import glob
import shutil
import numpy as np
from glob import glob
import os
import cv2
import xml.etree.ElementTree as ET
import argparse

import sys
sys.path.append('tools')
import _init_paths
from model.config import cfg
'''
the tool to product the format of voc (faster_rcnn)
1. 在用labelImg标注一批图片后，到项目根目录运行(把之前的样本都拿下来一起运行)：

python3.6 bin/parser_dataset.py --outdir=VOCdevkit2007/VOC2007 --xmldir=/path/to/img_xml --imagedir=path/to/img

C:\Users\Administrator\Desktop\VOCdevkit2007\VOC2007\JPEGImages

1. 在运行完以后会在你本地项目的根目录出现VOCdevkit2007文件夹, 
此时要在VOCdevkit2007/VOC2007/ImageSets/Main/目录下新建一个test.txt文件，
里面随便复制一些train.txt的内容就可以

'''
def build_voc_dirs(outdir):
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    ret = os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'ImageSets', 'Main')
    if mkdir is None:
      return ret
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    mkdir(os.path.join(outdir, 'SegmentationClass'))
    mkdir(os.path.join(outdir, 'SegmentationObject'))
    return ret

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='PARSER DATASET XML TOOL')

    parser.add_argument('--outdir', type=str, required=True, help="result out dir")
    parser.add_argument('--xmldir', type=str, required=True, help="xml labeled")
    parser.add_argument('--imagedir', type=str, required=True, help="images dir that will be train")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
  args = parse_args()

  # anns = glob(os.path.join('dataset/Annotations', '*.xml'))
  # _outdir = 'TEXTVOC/VOC2007'
  # _imgFromDir = '/Users/strucoder/Downloads/image/'

  anns = glob(os.path.join(args.xmldir, '*.xml'))
  _outdir = args.outdir
  _imgFromDir = args.imagedir
  dset = 'train'
  
  # class_sets = ('rider_chongfengyi_2018', 'rider_chongfengyind_2018', 'rider_majia_2018', 'rider_txue_2018', 'rider_box_2018', 'rider_boxwithele_2018', 'rider_txuewithele_2018', 'rider_toukui_2018')
  class_sets = cfg.CLASSES[1:]
  class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
  _dest_label_dir, _dest_img_dir, _dest_set_dir = build_voc_dirs(_outdir)

  fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets]
  ftrain = open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')
  
  for _file in anns:
    basename = os.path.basename(_file)
    filename, ext = os.path.splitext(basename)
    originImgPath = os.path.join(_imgFromDir, filename + '.png')
    imgExist = os.path.exists(originImgPath)
    # 如果对应的图片不存在则不拷贝
    if not imgExist:
      continue
    print('copy image: {}'.format(originImgPath))
    img = cv2.imread(originImgPath)
    cv2.imwrite(os.path.join(_dest_img_dir, filename + '.png'), img)

    ftrain.writelines(filename + '\n')
    # 解析xml文件
    tree = ET.parse(_file)
    objs = tree.findall('object')
    _cls_list = []
    for ix, obj in enumerate(objs):
      _cls = obj.find('name').text.lower().strip()
      _cls_list.append(_cls)
    cls_in_image = set(_cls_list)

    for cls_ in cls_in_image:
      if cls_ in class_sets:
        fs[class_sets_dict[cls_]].writelines(filename + ' 1\n')

    for cls_ in class_sets:
      if cls_ not in cls_in_image:
        fs[class_sets_dict[cls_]].writelines(filename + ' -1\n')

  (f.close() for f in fs)
  ftrain.close()
  shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'val.txt'))
  shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'trainval.txt'))
  
  for _file in anns:
    basename = os.path.basename(_file)

    # 如果对应的图片不存在则不copy此xml文件
    filename, ext = os.path.splitext(basename)
    originImgPath = os.path.join(_imgFromDir, filename + '.png')
    imgExist = os.path.exists(originImgPath)
    if not imgExist:
      continue
    shutil.copyfile(_file, os.path.join(_dest_label_dir, basename))

  for cls_ in class_sets:
    shutil.copyfile(os.path.join(_dest_set_dir, cls_ + '_train.txt'), os.path.join(_dest_set_dir, cls_ + '_trainval.txt'))
    shutil.copyfile(os.path.join(_dest_set_dir, cls_ + '_train.txt'), os.path.join(_dest_set_dir, cls_ + '_val.txt'))