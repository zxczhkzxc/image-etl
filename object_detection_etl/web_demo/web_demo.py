#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2019/4/29 14:06
@Author : zhanghongke
@Email : zhanghongke@dianwoda.com
@File : web_demo.py
@Desc:
有用的解决tf 资源不释放的方法:
,唯一的解决方法是使用进程并在计算后关闭它们 详见
https://codeday.me/bug/20180926/266737.html
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
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from flask import Flask, jsonify
import time
from nets.vgg16 import vgg16
import multiprocessing
font = cv2.FONT_HERSHEY_DUPLEX

# CLASSES= ('__background__', # always index 0
#        'rider_chongfengyi_2018', 'rider_chongfengyind_2018', 'rider_majia_2018', 'rider_txue_2018', 'rider_box_2018', 'rider_boxwithele_2018', 'rider_txuewithele_2018', 'rider_toukui_2018')
CLASSES = cfg.CLASSES
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

MODEL = 'vgg16_faster_rcnn_iter_70000.ckpt'
GPU_NUM = '/device:GPU:0'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xls', 'xlsx'])

app = Flask(__name__)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
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
    basepath = os.path.dirname(__file__)

    # cv2.imwrite('aaa.png', im)
    cv2.imwrite(os.path.join(basepath, 'static/images', 'aaa.png'), im)
    # vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    parser.add_argument('--img', type=str, required=True, help="Just Test image")
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

        demo(sess, net, im_name)


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        # load_tf_model('output/vgg16/voc_2007_trainval/default/', im_name)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        # cv2.imwrite(os.path.join(basepath, 'test.jpg'), img)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.png'), img)
        im_name=os.path.join(basepath, 'static/images', 'test.png')
        # args = ('process', lock)
        p = multiprocessing.Process(target=load_tf_model,args = ('output/vgg16/voc_2007_trainval/default/', im_name))
        p.start()
        p.join()
        # load_tf_model('output/vgg16/voc_2007_trainval/default/', im_name)

        return render_template('upload_ok.html', userinput=user_input, val1=time.time())

    return render_template('upload.html')
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # args = parse_args()

    # im_name = args.img
    # with tf.device(GPU_NUM):
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)