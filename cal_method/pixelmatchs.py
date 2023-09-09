from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

'''
利用像素之间的匹配来计算相似度
'''

def pixelmatchs(img_A, img_B):
    img_a = Image.open(img_A)
    img_b = Image.open(img_B)
    img_diff = Image.new("RGBA", img_a.size)
    # note how there is no need to specify dimensions
    mismatch = pixelmatch(img_a, img_b, img_diff, includeAA=True)
    # img_diff.save("diff.png")
    return mismatch

if __name__ == "__main__":
    img_a = "../ct/img_135_real_B.png"
    img_b = "../ct/img_135_fake_B.png"

    print(pixelmatchs(img_a, img_b))
    print(pixelmatchs(img_a, img_a))
