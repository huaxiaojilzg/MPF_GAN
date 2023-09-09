#图片分割
from PIL import Image
import os

filename = '../ct/epoch101_fake_B.png' #原图地址及名称
img = Image.open(filename)
size = img.size
# 准备将图片切割成32*32张小图片
weight = int(size[0] // 8)
height = int(size[1] // 8)

for j in range(8):
    for i in range(8):
        box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
        region = img.crop(box)
        region.save('ct_fenge/{}_{}.png'.format(j, i))

