import numpy as np
import cv2

from scipy.stats import wasserstein_distance

'''
越接近0越好 Wasserstein距离
'''
# 要知道自从2014年Ian Goodfellow提出以来，GAN就存在着训练困难、生成器和判别器的loss无法指示训练进程、生成样本缺乏多样性等问题。从那时起，很多论文都在尝试解决，但是效果不尽人意，比如最有名的一个改进DCGAN依靠的是对判别器和生成器的架构进行实验枚举，最终找到一组比较好的网络架构设置，但是实际上是治标不治本，没有彻底解决问题。而今天的主角Wasserstein GAN（下面简称WGAN）成功地做到了以下爆炸性的几点：
#
#     彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
#     基本解决了collapse mode的问题，确保了生成样本的多样性
#     训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，生成器产生的图像质量越高
#     以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到



def wd(img1_numpy, img2_numpy):
    img1_numpy = cv2.imread(img1_numpy)
    img2_numpy = cv2.imread(img2_numpy)

    w_sum = 0
    s, _, _ = img1_numpy.shape

    for i in range(s):
        for j in range(s):
            w_sum += wasserstein_distance(img1_numpy[i][j], img2_numpy[i][j])
    return w_sum


if __name__ == "__main__":
    img_ct_fake = "../ct/img_135_fake_B.png"
    img_ct_real = "../ct/img_135_real_B.png"
    img_pet_real = "../pet/test_pet.png"

    print(wd(img_ct_fake, img_ct_real))
    print(wd(img_ct_fake, img_pet_real))
    print(wd(img_ct_real, img_pet_real))
    print(wd(img_ct_real, img_ct_real))
