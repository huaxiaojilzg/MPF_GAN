import math
import cv2


def get_average(records):
    """
    平均值
    """
    return sum(records) / len(records)


def get_variance(records):
    """
    方差 反映一个数据集的离散程度
    """
    average = get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)


def get_standard_deviation(records):
    """
    标准差 == 均方差 反映一个数据集的离散程度
    """
    variance = get_variance(records)
    return math.sqrt(variance)


def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def get_mse_with_add_weight(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    records_real = cv2.imread(records_real)
    records_predict = cv2.imread(records_predict)

    if len(records_real) == len(records_predict):

        len_s = len(records_real)
        len_c = 1e-2

        zero_real_sum = 0
        zero_pred_sum = 0
        zero_sum = 0
        for i in range(len_s):
            for j in range(len_s):
                for k in range(3):
                    if records_real[i][j][k] == 0:
                        zero_real_sum += 1
                    if records_predict[i][j][k] == 0:
                        zero_pred_sum += 1
                    if not (records_real[i][j][k] == 0 and records_predict[i][j][k] == 0):
                        len_c += 1

        result = (sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len_c + abs(zero_real_sum - zero_pred_sum) / (zero_real_sum+zero_pred_sum))/len_s
        return sum(sum(result))
    else:
        return 0


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """

    # records_real = cv2.imread(records_real)
    # records_predict = cv2.imread(records_predict)

    mse = get_mse_with_add_weight(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return 0


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


if __name__ == '__main__':
    img1 = "../ct/epoch101_real_B.png"
    img2 = "../ct/epoch101_fake_B.png"
    img3 = "../pet/test_pet.png"

    img4 = "../ct/img_135_real_B.png"
    img5 = "../ct/img_135_fake_B.png"

    print(get_mse_with_add_weight(img1, img1))
    print('rmse',get_rmse(img1, img4))
    print(get_mse_with_add_weight(img1, img2))
    print(get_mse_with_add_weight(img1, img3))
    # print(get_mse(img1, img3))
    # print(get_mse(img2, img3))

    print(get_mse_with_add_weight(img4, img3))
    print(get_mse_with_add_weight(img4, img5))
    print(get_mse_with_add_weight(img4, img4))
