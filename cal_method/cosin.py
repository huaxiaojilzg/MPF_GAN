from PIL import Image
from numpy import average, linalg, dot
import os

# 余弦相似度
def gel_thumbnail(image, size=(256, 256), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image1, image2):
    image1 = Image.open(image1)
    image2 = Image.open(image2)

    image1 = gel_thumbnail(image1)
    image2 = gel_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    # res = 0
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)

    return res

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

        real_fake_ssim_data += image_similarity_vectors_via_numpy(real_ct, fake_ct)
        print('cal ' + str(i))
        i += 1

    print(real_fake_ssim_data / len(real_CT_list))


if __name__ == '__main__':
    # deal_train_result()
    img1 = r"D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2_5\test_init_2_merge_75\images\img_15828_real_B.png"
    img2 = r"D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2_5\test_init_2_merge_75\images\img_15828_fake_B.png"
    img3 = r"D:\mydata\GAN_learning\CycleGAN-and-pix2pix\results\train_init_2\test_init_2_merge_147\images\img_15828_fake_B.png"
    # img4 = "../pet/test_pet.png"

    ss1 = image_similarity_vectors_via_numpy(img1, img2)
    ss2 = image_similarity_vectors_via_numpy(img1, img3)
    # ss3 = calculate_ssim(img1, img4)
    print("ssim 1 real-fake", ss1)
    print("ssim 2 real-real", ss2)