# coding=gbk
import os
from skimage import io
import torch
from cellpose import dynamics
import tifffile
from super_params_set import own_parser

flow_args = own_parser.parse_args()
data_root = flow_args.data_path
flow_path = os.path.join(data_root, 'flow/')

train_root = os.path.join(data_root, 'train/')
test_root = os.path.join(data_root, 'test/')
train_coll = io.ImageCollection(train_root + 'input/*.tif')
train_coll_label = io.ImageCollection(train_root + 'annotate/*.tif')
test_coll = io.ImageCollection(test_root + 'input/*.tif')
test_coll_labels = io.ImageCollection(test_root + 'annotate/*.tif')

converted = 1


def label2flow(labels=None, file_names=None, save_path=None):
    """
    file_name: a list of names of the processing images, used for naming

    save_path: path for saving flows
    """
    os.makedirs(save_path, exist_ok=True)
    flow = dynamics.labels_to_flows(labels, use_gpu=True, device=torch.device('cuda'))
    for i in range(len(labels)):
        tifffile.imwrite(save_path + file_names[i], flow[i])


image_names_test = test_coll.files
image_names = train_coll.files

train_data = []
test_data = []
train_labels = []
test_labels = []
for i in range(len(train_coll.files)):
    train_data.append(train_coll[i])
    train_labels.append(train_coll_label[i])

for i in range(len(test_coll.files)):
    test_data.append(test_coll[i])
    test_labels.append(test_coll_labels[i])

file_names = []
test_file_names = []

if converted == 1:
    for name in image_names:
        file_name = os.path.split(name)[1]
        file_names.append(file_name)
    for test_name in image_names_test:
        test_file_name = os.path.split(test_name)[1]
        test_file_names.append(test_file_name)
    label2flow(train_labels, file_names=file_names, save_path=flow_path + 'train/')
    label2flow(test_labels, file_names=test_file_names, save_path=flow_path + 'test/')
