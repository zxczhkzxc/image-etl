#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2019/7/2 16:33
@Author : zhanghongke
@Email : zhanghongke@dianwoda.com
@File : remove_bad_img.py
@Desc: remove bad img for the result of label img
aim to use the voc to train
'''

import os
# source_dir='C:/Users/Administrator/Desktop/safe_toukui/'
source_dir='C:/Users/Administrator/Desktop/toukui_train/'
target_dir='C:/Users/Administrator/Desktop/toukui_bak/'
imagelist = os.listdir(source_dir)

print(imagelist)
name_list=[]
for imagename in imagelist:
    name=imagename.split(".")
    print("name",name[0])
    name_list.append(name[0])
arr_appear=dict((a,name_list.count(a)) for a in name_list)
print("arr_appear",arr_appear)
results=[]
count=0
for key,value in arr_appear.items():
    if value>1:
        count+=1
    else:
        results.append(key)
print("count",count)
print("len(results)",len(results))
print("results",results)
import os
import shutil
for slim_name in results:
    new_filename=source_dir+slim_name+'.png'
    target_name=target_dir+slim_name+'.png'
    print("new_filename",new_filename)
    #删除文件
    # os.remove(new_filename)
    # 移动文件
    shutil.move(new_filename, target_name)