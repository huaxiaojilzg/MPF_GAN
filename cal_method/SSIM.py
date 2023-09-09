import os

import cv2
import numpy as np

'''
值越大越好，越接近1  结构相似性，是一种衡量两幅图像相似度的指标
'''

# 结构相似度
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as
    img1, img2: [0, 255]
    '''

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
    elif img1.shape[2] == 1:
        return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def deal_train_result():
    result_dir = r'D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2_5\test_init_2_merge_75\images'
    img_lists = os.listdir(result_dir)

    real_PET_list=[]
    real_CT_list=[]
    fake_CT_list=[]

    for img in img_lists:

        if img.split('_')[2] == 'real':
            if img.split('_')[3] == 'A.png':
                real_PET_list.append(os.path.join(result_dir, img))
            else:
                real_CT_list.append(os.path.join(result_dir, img))
        else:
            fake_CT_list.append(os.path.join(result_dir, img))

    real_fake_ssim_data = 0

    i=1
    for real_ct, fake_ct in zip(real_CT_list, fake_CT_list):

        real_fake_ssim_data += calculate_ssim(real_ct, fake_ct)
        print('cal ' + str(i))
        i += 1




    print(real_fake_ssim_data / len(real_CT_list))




if __name__ == "__main__":
    img1 = r"D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2_5\test_init_2_merge_75\images\img_15828_real_B.png"
    img2 = r"D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2_5\test_init_2_merge_75\images\img_15828_fake_B.png"
    img3 = r"D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2\test_init_2_merge_147\images\img_15828_fake_B.png"
    # img4 = "../pet/test_pet.png"

    ss1 = calculate_ssim(img1, img2)
    ss2 = calculate_ssim(img1, img3)
    # ss3 = calculate_ssim(img1, img4)
    print("ssim 1 real-fake", ss1)
    print("ssim 2 real-real", ss2)
    # print("ssim 2 real-real", ss3)
    # deal_train_result()
