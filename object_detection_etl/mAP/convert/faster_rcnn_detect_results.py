#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2019/5/7 10:51
@Author : zhanghongke
@Email : zhanghongke@dianwoda.com
@File : faster_rcnn_detect_results.py
@Desc: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16

font = cv2.FONT_HERSHEY_DUPLEX

# CLASSES= ('__background__', # always index 0
#        'rider_chongfengyi_2018', 'rider_chongfengyind_2018', 'rider_majia_2018', 'rider_txue_2018', 'rider_box_2018', 'rider_boxwithele_2018', 'rider_txuewithele_2018', 'rider_toukui_2018')
CLASSES = cfg.CLASSES
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

MODEL = 'vgg16_faster_rcnn_iter_70000.ckpt'
GPU_NUM = '/device:GPU:0'


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    detections = []
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        for idx in inds:
            bbox = dets[idx, :4]
            score = dets[idx, -1]
            x = int(bbox[0])
            y = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cv2.rectangle(im, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, '{}:{:.3f}'.format(cls, score), (x + 6, y + 20), font, 0.8, (255, 0, 0), 2)
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # detections.append([left, top, right, bottom, cls, score])
            detections.append([x, y, x2, y2, cls, score])
    return detections
    # detections = []
    # cv2.imwrite('aaa.png', im)
    # vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    # parser.add_argument('--img', type=str, required=True, help="Just Test image")
    parser.add_argument('--model', type=str, required=True, help="trained model dir")
    # parser.add_argument('--ckpt_dir', type=str, required=True, help="ckpt dir")
    parser.add_argument('--check', type=str, required=True, help="need to check image dir")
    parser.add_argument('--output', type=str, required=True, help="output dir")
    # parser.add_argument('--csv_path', type=str, required=True, help="csv_path")
    # parser.add_argument('--day', type=str, required=True, help="day")
    # parser.add_argument('--child_dir', type=str, required=True, help="child_dir")
    # parser.add_argument('--all_day', type=str, required=True, help="yes is a day, no is not")
    args = parser.parse_args()

    return args


def load_tf_model(ckpt_dir, im_name):
    net = vgg16()
    net.create_architecture("TEST", len(CLASSES), tag='default', anchor_scales=[8, 16, 32])
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # sess = tf.Session(config=config)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        # saver.restore(sess, ckpt.model_checkpoint_path)

        ckpt_path = os.path.join(ckpt_dir, MODEL)
        saver.restore(sess, ckpt_path)

        detections=demo(sess, net, im_name)
    return detections


if __name__ == '__main__':
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    args = parse_args()
    model_dir = args.model
    check = args.check
    # csv_path = args.csv_path
    output = args.output
    # day = args.day
    # child_dir = args.child_dir
    # isAllDay = args.all_day

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    check_images_path = check
    # check_images_path = os.path.join(check, day, child_dir)
    # savefile = os.path.join(output, 'body_{}.csv'.format(child_dir))
    # im_name = args.img
    # with tf.device(GPU_NUM):
    result_detections = []
    result_images = []

    for root, dirs, files in os.walk(check_images_path):
        for name in files:
            tf.reset_default_graph()
            check_img_path = os.path.join(root, name)
            if os.path.splitext(check_img_path)[1] == '.png':
                _name = os.path.basename(check_img_path)
                lastIdx = _name.rindex('-')
                # __name = _name[0:lastIdx]
                # 保存工单 1-199153-20180620Q17506050.png
                # work_order_num[_name] = _name[lastIdx + 1: (len(_name) - 4)]
                # all_check_imgs[_name] = cv2.imread(check_img_path)
                detections=load_tf_model(model_dir, check_img_path)
                result_images.append(name)
                result_detections.append(detections)
                # load_tf_model('output/vgg16/voc_2007_trainval/default/', check_img_path)
    import datetime
    output_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    pti01_output_path = output + 'voc_vgg_{}.txt'.format(output_version)
    print('Saving in ', pti01_output_path)

    with open(pti01_output_path, 'w') as output_f:
        for index, image_filename in enumerate(result_images):
            detections_string = ''
            for d in result_detections[index]:
                # <left> <top> <right> <bottom> <class_id> <confidence>
                detections_string += ' {},{},{},{},{},{}'.format(d[0], d[1], d[2], d[3], d[4], d[5])

            output_f.write('{}{}\n'.format(image_filename, detections_string))