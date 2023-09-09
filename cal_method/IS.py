# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
import cv2

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=100):
    p_yx = cv2.imread(p_yx)
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


# conditional probabilities for low quality images
# p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
# img_ct_fake = cv2.imread("../ct/img_135_fake_B.png")
# img_ct_real = cv2.imread("../ct/img_135_real_B.png")
#
# print(calculate_inception_score(img_ct_fake))
print(calculate_inception_score(r"images_ker_size=3\epoch001_fake_B.png"))

