
import os
from cellpose import utils, models
import torch
import skimage.io as io
from super_params_set import parser

train_args = parser.parse_args()

data_root = train_args.data_path
save_path = train_args.model_save
test_while_train_path = train_args.test_while_train_path

os.makedirs(save_path, exist_ok=True)

flow_path = data_root + 'flow/'
train_root = os.path.join(data_root, 'train/')
test_root = os.path.join(data_root, 'test/')
ntest = len(os.listdir(test_root + 'input/'))
ntrain = len(os.listdir(train_root + 'input/'))

pretrained_model = train_args.pretrained_model_path


def train_cellpose(train_data=None, train_labels=None,
                   test_data=None, test_labels=None, pretrained_model=None):
    cellpose_model = models.CellposeModel(device=torch.device('cuda'),
                                          nchan=2,
                                          style_on=train_args.style_on,
                                          residual_on=train_args.residual_on,
                                          concatenation=train_args.concatenation,
                                          model_name=train_args.model_name,
                                          diam_mean=40,
                                          pretrained_model=pretrained_model)

    cpmodel_path = cellpose_model.train(train_data=train_data, train_labels=train_labels,
                                        test_data=test_data, test_labels=test_labels,
                                        test_while_train_root=test_while_train_path,
                                        start_epoch=train_args.start_epoch,
                                        n_epochs=train_args.n_epochs,
                                        learning_rate=train_args.learning_rate,
                                        weight_decay=train_args.weight_decay,
                                        channels=[0, 0],
                                        batch_size=train_args.batch_size,
                                        save_path=os.path.realpath(save_path),
                                        save_every=train_args.save_every,
                                        model_name=train_args.model_name)


def train_size(train_data=None, train_labels=None,
               test_data=None, test_labels=None, pretrained_model=None):
    cpmodel = models.CellposeModel(device=torch.device('cuda'),
                                   nchan=2,
                                   model_name='glomeruli_size',
                                   pretrained_model=pretrained_model)

    sz_model = models.SizeModel(cp_model=cpmodel)

    sz_model.train(train_data=train_data, train_labels=train_labels,
                   test_data=test_data, test_labels=test_labels,
                   channels=[0, 0],
                   n_epochs=50)


train_names = os.listdir(train_root + 'input/')
test_names = os.listdir(test_root + 'input/')
train_flow_names = os.listdir(flow_path + 'train/')
test_flow_names = os.listdir(flow_path + 'test/')

train_data = []
test_data = []
train_labels = []
test_labels = []
for i in range(ntrain):
    train_data.append(io.imread(train_root + '/input/' + train_names[i]))
    label = io.imread(flow_path + '/train/' + train_flow_names[i])
    label = label.transpose(2, 0, 1)
    train_labels.append(label)

for i in range(ntest):
    test_data.append(io.imread(test_root + '/input/' + test_names[i]))
    label = io.imread(flow_path + '/test/' + test_flow_names[i])
    label = label.transpose(2, 0, 1)
    test_labels.append(label)

train_cellpose(train_data=train_data, train_labels=train_labels,
               test_data=test_data, test_labels=test_labels,
               pretrained_model=pretrained_model)

# train_size(train_data, train_labels, test_data, test_labels, pretrained_model=pretrained_model)
