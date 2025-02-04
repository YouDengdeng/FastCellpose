# FastCellpose

## Citations

Han Y, Zhang Z, Li Y, et al. FastCellpose: A Fast and Accurate Deep-Learning Framework for Segmentation of All Glomeruli in Mouse Whole-Kidney Microscopic Optical Images[J]. Cells, 2023, 12(23): 2753.

## Dependencies

FastCellpose uses nearly the same environment configuration as Cellpose project, and has been heavily tested on Windows 10 system. Here are the packages they rely on(which can be  automatically installed with conda/pip if missing):

- [pytorch](https://pytorch.org/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [skimage](https://scikit-image.org/)
- [cv2](https://opencv.org/releases/)

You can check [Cellpose/README.md](https://github.com/MouseLand/cellpose/blob/main/README.md) for more Cellpose's local installation details.

## Demo

Before you start your own data training process, we offer a demo project to ensure you have configured the proper environment. 

*NOTICE*
*It's worth noticing that for different segmentation tasks, the hyparameters need to be finetuned to achieve better results.*

In folder *demo_infer* we prepare a fine-trained model and 2 2048×2048-pixel images. You can run *2_finalspeed_whole_inference.py* and check the inputs/results in:

```
└─demo_infer
    ├─inference_out
```

## Own data training

FastCellpose requires a sufficient amount of data  to complete its supervised training process. Here we introduce the folders and process to deal with your data(every folder has a demo image, please check them out if you have any problem).

1. Split your data into train and test set, then move them to the corresponding folders: *origin* and *whole_mask*.

2. Transfer your images and GT into small-size patches, and move them to the corresponding folders: *input* and *mask*. Remember to save some data under folder *test_while_train* to track the training results.

3. Transfer your mask patches into annotated form, which means background's gray value is 0, while different segmented objects' are 1,2,etc. 

4. Transfer the annotated form into flow using *0_flow_production.py* .

5. Start your training process using *1_train_glomeruli.py*.

6. Comment out the demo part and start your inference using *2_finalspeed_whole_inference.py*.
   
   **we complete 1-3 using Matlab with the code attatched in folder *preprocess_matlab*. Before you start 4-6, it's recommended to skim over *super_params_set.py* and set the correct super params including data path, training/testing params and more.**

```
└─data
    ├─flow
    │  ├─test
    │  │      
    │  └─train
    ├─test
    │  ├─annotate
    │  │      
    │  ├─input
    │  │      
    │  ├─mask
    │  │      
    │  ├─origin
    │  │      
    │  ├─whole_mask
    ├─test_while_train
    │  ├─input
    │  │      
    │  ├─mask
    └─train
        ├─annotate
        │      
        ├─input
        │      
        ├─mask
        │      
        ├─origin
        │      
        └─whole_mask
```

## Advanced model settings

### LR iteration

Change in *core.py* line 851

### base feature channels num of U-Net

Change in *core.py*. there are [16, 32, 64, 128] and [32, 64, 128, 256] to choose in python class: UnetModel

### conv num in every encoder/decoder layer of U-Net

Change in *resnet_torch.py*. There are 2 or 4 to choose by commenting or uncommenting out those python class: convup, convdown, resup and resdown
