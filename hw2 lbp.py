from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# settings for LBP
radius = 1  # LBP算法中範圍半徑的取值
n_points = 8 * radius # 領域像素點數
# 讀取圖像
image = cv2.imread('test1.jpg')
#顯示到plt中，需要從BGR轉化到RGB，若是cv2.imshow(win_name, image)，則不需要轉化
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(111)
plt.imshow(image1)

image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
plt.subplot(111)
plt.imshow(image, plt.cm.gray)
lbp = local_binary_pattern(image, n_points, radius)
plt.subplot(111)
plt.imshow(lbp, plt.cm.gray)
'''
edges = filters.sobel(image)
plt.subplot(111)
plt.imshow(edges, plt.cm.gray)
'''

