# TF_Segmentation
 Tensorflow 2.2 Segmentation learning tests

All models are trained on ADE20K dataset

Classes: 2 (BG + Wall)

DataGenerator.py
    - separate module for input pipeline
    - modified random crop so that it generates the same crop on both image and mask

Unet_512_augmented.py
    - Loss: SparseCategoricalCrossentropy
    - 512 px input
    - working version of a Unet model
    - overfits in aprox 20 batches
    - Total params: 1,962,642

Unet_512_augmented_v2.py 
    - lowered number of parameters
    - Total params: 491,146

Unet_512_augmented_v3.py
    - 512 px, seed = 15
    - added new DataGenerator
    - best performance so far for a Unet model
    - started overfitting after epoch 31

        loss: 0.2928 -     accuracy: 0.8732 -     iou: 0.7305 -     dsc: 0.7611 -     tversky: 0.7386  
    val_loss: 0.2998 - val_accuracy: 0.8704 - val_iou: 0.7244 - val_dsc: 0.7542 - val_tversky: 0.7246 
    lr: 1.0000e-04

Unet_512_edges
    - not working
    - loss function not helping this unbalanced task

Fast SCNN
    - based on https://github.com/templeblock/fast-scnn-keras/blob/master/model/model.py

UNet ResNet 
    - implementation from https://github.com/foobar167/articles/blob/master/Machine_Learning/code_examples/deep_residual_unet_segmentation.ipynb
