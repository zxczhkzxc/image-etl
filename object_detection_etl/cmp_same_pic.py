#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2019/5/30 11:05
@Author : zhanghongke
@Email : zhanghongke@dianwoda.com
@File : cmp_same_pic.py
@Desc: 
'''
import cv2
import numpy as np

# import os

file1 = "1.png"
file2 = "2.png"

image1 = cv2.imread(file1)
image2 = cv2.imread(file2)
print(image1.shape)
print(image2.shape)
size1 = image1.shape
width1 = size1[0]
height1 = size1[1]

size2 = image2.shape
width2 = size2[0]
height2 = size2[1]

height=int((height2+height1)/2)
width=int((width1+width2)/2)
# img1 = image1.reshape(-1, height*0.5, 3)
# img2 = image2.reshape(-1, height*0.5, 3)
# 也就是说resize 是默认的先fx轴，后fy轴，也就是先宽后高
img1 = cv2.resize(image1, (height,width ),interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(image2, (height,width ),interpolation=cv2.INTER_CUBIC)

print(img1.shape)
print(img2.shape)
difference = cv2.subtract(img1, img2)
# print("difference",difference)
result = not np.any(difference)  # if difference is all zeros it will return False
print("result",result)

if result is True:
    print("两张图片一样")
else:
    cv2.imwrite("result.jpg", difference)
    print("两张图片不一样")


from skimage.measure import compare_ssim
import cv2

class CompareImage():

    def compare_image(self, imageA, imageB):

        # imageA = cv2.imread(path_image1)
        # imageB = cv2.imread(path_image2)

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        print("SSIM: {}".format(score))
        return score


compare_image = CompareImage()
# compare_image.compare_image(file1, file2)
compare_image.compare_image(img1, img2)

