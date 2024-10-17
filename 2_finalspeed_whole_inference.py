# -*- coding: gbk -*-

import logging
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from cellpose import models
import cv2
import numpy as np
from tqdm import trange
from super_params_set import parser

from cellpose import transforms, utils, dynamics
from cellpose.models import models_logger
from score_3 import dice_coef, iou_score

test_args = parser.parse_args()

# for your own dataset inference,run :

# test_img_path = test_args.inf_img_path
# mask_img_path = test_args.inf_GT_path
# results_path = test_args.results_path
# style_on = test_args.style_on
# residual_on = test_args.residual_on
# concatenation = test_args.concatenation
# model_name = test_args.model_name
# inference_model_name = test_args.inference_model_name

# for DEMONSTRATION, plz run instead:

# test_img_path = r'H:\Code\Python_code\Fast_Cellpose_prj\demo_infer\img/'
# mask_img_path = r'H:\Code\Python_code\Fast_Cellpose_prj\demo_infer\mask/'
# # mask_img_path = None
# results_path = r'H:\Code\Python_code\Fast_Cellpose_prj\demo_infer\inference_out/'
# style_on = False
# residual_on = False
# concatenation = False
# model_name = 'demoglo_nbase=32_conv=2'
# inference_model_name = r'H:\Code\Python_code\Fast_Cellpose_prj\demo_infer\demoglo_nbase=32_conv=2.pth'

test_img_path = r'.\demo_infer\img/'
mask_img_path = r'.\demo_infer\mask/'
# mask_img_path = None
results_path = r'.\demo_infer\inference_out/'
style_on = False
residual_on = False
concatenation = False
model_name = 'demoglo_nbase=32_conv=2'
inference_model_name = r'.\demo_infer\demoglo_nbase=32_conv=2.pth'


log_final = os.path.join(results_path, 'log_prediction.txt')
cellpose_model = models.CellposeModel(device=torch.device('cuda'),
                                      nchan=2,
                                      style_on=style_on,
                                      residual_on=residual_on,
                                      concatenation=concatenation,
                                      model_name=model_name,
                                      diam_mean=40,
                                      pretrained_model=inference_model_name)

patch_size = test_args.patch_size
patch_edge = test_args.patch_edge
valid_patch = patch_size - 2 * patch_edge

os.makedirs(results_path, exist_ok=True)


def fulfill(inp):

    data = []
    [w, h, channel] = inp.shape
    r = (np.ceil(w / valid_patch) * valid_patch).astype(int) + patch_edge * 2
    c = (np.ceil(h / valid_patch) * valid_patch).astype(int) + patch_edge * 2
    in_img = np.zeros((r, c, channel))
    in_img[patch_edge:patch_edge + w, patch_edge:patch_edge + h, :] = inp
    in_img = in_img.astype('float32')
    for m in range(0, r - patch_size + 1, valid_patch):
        for n in range(0, c - patch_size + 1, valid_patch):
            image = in_img[m:m + patch_size, n:n + patch_size, :]
            if image.mean().mean() < 0.12:
                image = np.zeros(image.shape)
            data.append(image)
    return [data, r, c, w, h]


def predict_and_evaluate(self, bsize, down_size=1, niter=50):
    test_img_name_list = os.listdir(test_img_path)
    length = len(test_img_name_list)
    dice = []
    iou = []
    mk1_mean = []
    mk2_mean = []
    for i in range(0, length):
        print("processing %d img" % i)
        inp = cv2.imread(test_img_path + test_img_name_list[i])
        [data_list, r, c, w, h] = fulfill(inp)

        test_img = data_list
        n_img = len(data_list)
        shape = test_img[0].shape

        test_img = [transforms.convert_image(img, channels=[0, 0]) for img in test_img]
        tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
        iterator = trange(n_img, file=tqdm_out) if n_img > 1 else range(n_img)
        dP = np.zeros((2, n_img, shape[0], shape[1]), np.float32)
        cellprob = np.zeros((n_img, shape[0], shape[1]), np.float32)

        mk1_o = time.time()
        for j in iterator:
            if j % bsize == 0:
                startpt = j
                endpt = j + bsize
                if endpt >= n_img:
                    endpt = n_img
                img = np.asarray(test_img[startpt:endpt])
                imgs = np.transpose(img, (0, 3, 1, 2))
                X = self._to_device(imgs)
                self.net.eval()
                with torch.no_grad():
                    y = self.net(X)[0]
                    # y = self.net(X)
                del X
                y = self._from_device(y)
                # transpose so channels axis is last again
                y = np.transpose(y, (0, 2, 3, 1))
                cellprob[startpt:endpt] = y[:, :, :, 2]  # y:256X256X3
                dP[:, startpt:endpt] = y[:, :, :, :2].transpose((3, 0, 1, 2))  # dP:2X256X256X256
        mk1_e = time.time()
        real_shape = [r - 2 * patch_edge, c - 2 * patch_edge]
        mk2_o = time.time()
        out_dp_whole = np.zeros((2, real_shape[0], real_shape[1]), np.float32)
        cellprob_whole = np.zeros((real_shape[0], real_shape[1]), np.float32)
        index = 0
        for m in range(0, real_shape[0] - valid_patch + 1, valid_patch):
            for n in range(0, real_shape[1] - valid_patch + 1, valid_patch):
                out_dP = dP[:, index]  # 2X256X256
                out_cellprob = cellprob[index]
                out_dp_whole[:, m:m + valid_patch, n:n + valid_patch] = out_dP[:, patch_edge:patch_edge + valid_patch,
                                                                        patch_edge:patch_edge + valid_patch]
                cellprob_whole[m:m + valid_patch, n:n + valid_patch] = out_cellprob[patch_edge:patch_edge + valid_patch,
                                                                       patch_edge:patch_edge + valid_patch]
                index += 1

        whole_down_dp = np.zeros((2, real_shape[0] // down_size, real_shape[1] // down_size), np.float32)
        whole_down_dp[0, :, :] = transforms.resize_image(out_dp_whole[0, :, :], real_shape[0] // down_size,
                                                         real_shape[1] // down_size, interpolation=cv2.INTER_CUBIC)
        whole_down_dp[1, :, :] = transforms.resize_image(out_dp_whole[1, :, :], real_shape[0] // down_size,
                                                         real_shape[1] // down_size, interpolation=cv2.INTER_CUBIC)
        whole_down_cellprob = transforms.resize_image(cellprob_whole, real_shape[0] // down_size,
                                                      real_shape[1] // down_size,
                                                      interpolation=cv2.INTER_CUBIC)
        outputs = dynamics.compute_masks(whole_down_dp, whole_down_cellprob, niter=niter,
                                         cellprob_threshold=0,
                                         flow_threshold=0.4, interp=True, resize=real_shape,
                                         use_gpu=self.gpu, device=self.device)
        mask = outputs[0]
        pre_out = mask[0:w, 0:h]
        pre_out[pre_out > 0] = 1
        pre_out = np.array(pre_out, dtype=np.uint8)
        mk2_e = time.time()

        mk1 = mk1_e - mk1_o
        mk2 = mk2_e - mk2_o

        cv2.imwrite(os.path.join(results_path, test_img_name_list[i]), pre_out * 255)

        print("mk_getflow = %s" % mk1, "\t")
        print("mk_getmask = %s" % mk2, "\t")
        mk1_mean.append(mk1)
        mk2_mean.append(mk2)
        if i == 0:
            f = open(log_final, 'w')
        else:
            f = open(log_final, 'a')
        if mask_img_path is not None:
            mask_name = os.path.join(mask_img_path, test_img_name_list[i])
            test_lbl = cv2.imread(mask_name, 0)
            test_lbl[test_lbl > 0] = 1
            d2 = dice_coef(pre_out, test_lbl)
            i2 = iou_score(pre_out, test_lbl)

            dice.append(d2)
            iou.append(i2)
            f.write("%d : dice: %4f, iou: %4f\n" % (i, d2, i2))
        else:
            f.write("no GT_mask\n")
        f.write("mk_getflow = %4f\t" % mk1)
        f.write("mk_getmask = %4f\n" % mk2)
        f.close()

        print('inference ends')

    f = open(log_final, 'a')
    mk1_std = np.std(mk1_mean[1:])
    mk2_std = np.std(mk2_mean[1:])

    mk1_mean_time = np.mean(mk1_mean[1:])
    mk2_mean_time = np.mean(mk2_mean[1:])
    if mask_img_path is not None:
        dice_std = np.std(dice)
        iou_std = np.std(iou)
        mdice = np.mean(dice)
        miou = np.mean(iou)

        f.write("Average: dice: %4f, iou: %4f\n" % (mdice, miou))
        f.write("Average: dice_std = %4f" % dice_std)
        f.write("Average: iou_std = %4f\n" % iou_std)
    else:
        print("no GT mask")
    f.write("Average: mk_getflow = %4f" % mk1_mean_time)
    f.write("Average: mk_getmask = %4f\n" % mk2_mean_time)
    f.write("Average: mk1_std = %4f" % mk1_std)
    f.write("Average: mk2_std = %4f" % mk2_std)


predict_and_evaluate(self=cellpose_model, bsize=test_args.inf_batch_size, down_size=test_args.grad_track_down,
                     niter=test_args.grad_track_niter)
