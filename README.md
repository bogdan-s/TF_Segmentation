# TF_Segmentation
 Tensorflow 2.2 Segmentation learning tests

All models are trained on ADE20K dataset

Classes: 2 (BG + Wall)

Unet_512_augmented.py
    - Loss: SparseCategoricalCrossentropy
    - 512 px input
    - working version of a Unet model
    - overfits in aprox 20 batches
    - Total params: 1,962,642



Unet_512_augmented_v2.py 
    - lowered number of parameters
    - Total params: 491,146
 
Unet_512_edges
    - not working
    - loss function not helping this unbalanced task

Fast SCNN
    - based on https://github.com/templeblock/fast-scnn-keras/blob/master/model/model.py

UNet ResNet 
    - implementation from https://github.com/foobar167/articles/blob/master/Machine_Learning/code_examples/deep_residual_unet_segmentation.ipynb
