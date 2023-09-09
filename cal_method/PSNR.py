'''
值越大越好！！！！！越接近100 峰值信噪比
'''
import cv2
import numpy as np
import math
import os



def psnr1(img1, img2):

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def main():
    img1 = "../ct/epoch101_real_B.png"
    img2 = "../ct/epoch101_fake_B.png"
    img3 = "../pet/test_pet.png"
    # img4 = "ct_real/img_135_real_B.png"

    # img1 = cv2.imread(img1)
    # img2 = cv2.imread(img2)
    # img3 = cv2.imread(img3)

    print("p1 redl-fake:", psnr1(img1, img2))
    print("p1 redl-real:",psnr1(img1, img1))
    print("p1 ct-pet:",psnr1(img1, img3))
    print("p2 ct-pet:",psnr1(img2, img3))

    # print("p2 redl-fake:", psnr2(img1, img2))
    # print("p2 redl-real:", psnr2(img1, img3))


if __name__ == "__main__":
    main()
