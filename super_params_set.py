import argparse
import torch

# Params in preprocess and flow production
parser = argparse.ArgumentParser(description='cellpose parameters')
# preprocess settings : patch_size, patch_overlap_size
patch_args = parser.add_argument_group("patch arguments")
patch_args.add_argument('--patch_size', default=256, help='patch size')
patch_args.add_argument('--patch_edge', default=64, help='patch overlap size')

# data path params
data_args = parser.add_argument_group("data arguments")
data_args.add_argument('--data_path', default=r'H:\Code\Python_code\Fast_Cellpose_prj\data/', help='user data path')
data_args.add_argument('--model_save', default=r'H:\Code\Python_code\Fast_Cellpose_prj\model\model_deconv_16_2/', help='user model path')
data_args.add_argument('--test_while_train_path', default=r'H:\Code\Python_code\Fast_Cellpose_prj\data\test_while_train/', help='data to track your training results')

# Common params for model in training and inference
common_args = parser.add_argument_group("common arguments")
common_args.add_argument('--device', default=torch.device('cuda'), help='gpu used to train')
common_args.add_argument('--style_on', default=False, help='use style vector in network')
common_args.add_argument('--residual_on', default=False, help='use ResBlock in network')
common_args.add_argument('--concatenation', default=False, help='False means pixel-wise summation in skip connection')
common_args.add_argument('--model_name', default='glo_nbase=16_conv=2', help='name your model')

# train settings
train_set_args = parser.add_argument_group("train setting arguments")
train_set_args.add_argument('--start_epoch', default=0, help='epoch start, facilitate your successive training process')
train_set_args.add_argument('--pretrained_model_path', default=None, help='pretrained_model path, default to be None')
train_set_args.add_argument('--n_epochs', default=100, help='total epoch nums')
train_set_args.add_argument('--learning_rate', default=0.2, help='Initial  lr, iteration strategy can change in:core.py line 851')
train_set_args.add_argument('--weight_decay', default=0.00001, help='for regularization')
train_set_args.add_argument('--batch_size', default=4, help='batch_size in training process')
train_set_args.add_argument('--save_every', default=1, help='save your model every xx epochs')

# customized test settings
test_set_args = parser.add_argument_group("inference setting arguments")
test_set_args.add_argument('--inf_img_path', default=r'H:\Code\Python_code\Fast_Cellpose_prj\data\test\origin/', help='imgs for inference')
test_set_args.add_argument('--inf_GT_path', default=r'H:\Code\Python_code\Fast_Cellpose_prj\data\test\whole_mask/', help='GT_masks for inference test')
# test_set_args.add_argument('--inf_GT_path', default=None, help='GT_masks for inference test, set None to purely inference without GT_mask')

test_set_args.add_argument('--results_path', default=r'H:\Code\Python_code\Fast_Cellpose_prj\data\test\whole_predict', help='to save the inference output')
test_set_args.add_argument('--inference_model_name', default=r'H:\Code\Python_code\Fast_Cellpose_prj\model\model_deconv_16_2\models\glo_nbase=16_conv=2_epoch_9.pth', help='model used for inference')
test_set_args.add_argument('--inf_batch_size', default=16, help='inference xx imgs at the same time')
test_set_args.add_argument('--grad_track_down', default=2, help='downsample before gradient tracking process')
test_set_args.add_argument('--grad_track_niter', default=50, help='gradient track iterations ')

