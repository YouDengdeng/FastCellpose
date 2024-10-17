import numpy as np
import os
import tifffile as tiff

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def dice_coef(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def iou_score(output, target):
    smooth = 1e-5

    # output_ = output > 0.5
    # target_ = target > 0.5
    # intersection = (output_ & target_).sum()
    # union = (output_ | target_).sum()

    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def sensitivity(output, target):
    smooth = 1e-5

    intersection = (output * target).sum()

    return (intersection + smooth) / (target.sum() + smooth)


def ppv(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()

    return (intersection + smooth) / (output.sum() + smooth)
    # 计算了dice、iou、sensitivity、ppv


if __name__ == "__main__":

    mask_path = ''
    log_predict = ''
    gt_path = ''

    output = []
    target = []
    dice = []
    iou = []
    sen = []
    ppv_score = []
    i = 0
    for mask_name, gt_name in zip(os.listdir(mask_path), os.listdir(gt_path)):
        output = tiff.imread(mask_path + mask_name)
        output[output > 0] = 1
        target = tiff.imread(gt_path + gt_name)
        target[target > 0] = 1
        dice.append(dice_coef(output, target))
        iou.append(iou_score(output, target))
        sen.append(sensitivity(output, target))
        ppv_score.append(ppv(output, target))
        if i == 0:
            f = open(log_predict, 'w')
        else:
            f = open(log_predict, 'a')
        f.write("%s\t" % mask_name)
        f.write("dice: %4f, iou score: %4f, sensitivity: %4f, ppv: %4f\n" % (dice[i], iou[i], sen[i], ppv_score[i]))
        f.close()
        i += 1

    dice = np.mean(dice)
    iou = np.mean(iou)
    sen = np.mean(sen)
    ppv_score = np.mean(ppv_score)
    print("dice: %4f, iou score: %4f, sensitivity: %4f, ppv: %4f" % (dice, iou, sen, ppv_score))
    f = open(log_predict, 'a')
    f.write("Average\t")
    f.write("dice: %4f, iou score: %4f, sensitivity: %4f, ppv: %4f" % (dice, iou, sen, ppv_score))
