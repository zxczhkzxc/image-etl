#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2019/6/5 17:30
@Author : zhanghongke
@Email : zhanghongke@dianwoda.com
@File : ctw_to_icdar.py
@Desc: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
# import settings
import shutil

# from pythonapi import anno_tools, common_tools
from pythonapi.anno_tools import *
from pythonapi.common_tools import *
# from pythonapi.anno_tools import *
data_dir='E:\data\ctw-annotations\\'
target_dir='E:\data\ctw-annotations-target\\'
print("123")
import pprint
# DATA_LIST=settings.DATA_LIST
# TRAIN=settings.TRAIN
# VAL=settings.VAL
# TEST_CLASSIFICATION=settings.TEST_CLASSIFICATION
# TEST_CLASSIFICATION_GT=settings.TEST_CLASSIFICATION_GT
# TEST_DETECTION_GT=settings.TEST_DETECTION_GT

DATA_LIST=target_dir+'info.json'
TRAIN=target_dir+'train.jsonl'
VAL=target_dir+'val.jsonl'
TEST_CLASSIFICATION=target_dir+'test_cls.jsonl'
TEST_CLASSIFICATION_GT=target_dir+'test_cls.gt.jsonl'
TEST_DETECTION_GT=target_dir+'test_det.gt.jsonl'

def main():
    # load from downloads
    # with open('../data/annotations/downloads/info.json') as f:
    with open(data_dir+'info.json') as f:
        full_data_list = json.load(f)
    with open(data_dir+'val.jsonl') as f:
        val = f.read().splitlines()
    # print(full_data_list)
    pprint.pprint(full_data_list['train'][0])
    # make infomation of data list
    data_list = {
        'train': full_data_list['train'],
        'val': [],
        'test_cls': full_data_list['val'],
        'test_det': full_data_list['val'],
    }
    with open(DATA_LIST, 'w') as f:
        json.dump(data_list, f, indent=2)

    # copy training set
    # shutil.copy(data_dir+'train.jsonl', TRAIN)

    # create empty validation set
    with open(VAL, 'w') as f:
        pass

    # create testing set for classification
    with open(TEST_CLASSIFICATION, 'w') as f, open(TEST_CLASSIFICATION_GT, 'w') as fgt:
        for line in val:
            anno = json.loads(line.strip())
            proposals = []
            gt = []

            for char in each_char(anno):
                if not char['is_chinese']:
                    continue
                proposals.append({'adjusted_bbox': char['adjusted_bbox'], 'polygon': char['polygon']})
                gt.append({'text': char['text'], 'attributes': char['attributes'], 'size': char['adjusted_bbox'][-2:]})
            anno.pop('annotations')
            anno.pop('ignore')
            anno['proposals'] = proposals
            print("anno", anno)
            f.write(to_jsonl(anno))
            f.write('\n')
            anno.pop('proposals')
            anno['ground_truth'] = gt
            fgt.write(to_jsonl(anno))
            fgt.write('\n')

    # create testing set for detection
    shutil.copy(data_dir+'val.jsonl', TEST_DETECTION_GT)


if __name__ == '__main__':
    main()
