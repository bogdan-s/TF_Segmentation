

import os
import sys
from pathlib import Path
import datetime

from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D

import tensorflow_addons as tfa


# # Device Compute Precision - mixed precision
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__, end='\n\n')
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
# param
IMG_SIZE_Before_Crop = 512 #150 for 128 final image
IMG_SIZE = 512
BATCH_SIZE = 6
OUTPUT_CHANNELS = 2
EPOCHS = 60
away_from_computer = True  # to show or not predictions between batches
save_model_for_inference = False # to save or not the model for inference
SEED = 15

# dataset location
Train_Images_Path = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/"
Val_Images_Path =  "D:/Python/DataSets/ADE20K_Filtered/Validation/Images/0/"

# similar to glob but with tensorflow
train_imgs = tf.data.Dataset.list_files(Train_Images_Path + "*.jpg", shuffle=False)
val_imgs = tf.data.Dataset.list_files(Val_Images_Path + "*.jpg", shuffle=False)

# @tf.function
def parse_image(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(image, channels=3)
    
    mask_path = tf.strings.regex_replace(img_path, "Images", "New_Masks")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_seg.png")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=0, dtype=tf.dtypes.uint8)
    print(mask.shape)
    
    return {'image': image, 'segmentation_mask' : mask}

train_set = train_imgs.map(parse_image, num_parallel_calls=AUTOTUNE)

test_set = val_imgs.map(parse_image, num_parallel_calls=AUTOTUNE)
dataset = {"train": train_set, "test": test_set}

print(dataset.keys())

# @tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE_Before_Crop, IMG_SIZE_Before_Crop), method='area')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE_Before_Crop, IMG_SIZE_Before_Crop), method='area') #, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    
    input_image = input_image / 255
    input_mask = tf.image.rgb_to_grayscale(input_mask)
    input_mask = tf.floor(input_mask / 255 + 0.5)
    # print("Shape: {}".format(input_image.shape))
    # print("Shape: {}".format(input_mask.shape))
    return input_image, input_mask

# @tf.function
def train_random_crop(crop_image, crop_mask) -> tuple:
    print("Shape image before crop: {}".format(crop_image.shape))
    print("Shape mask before crop: {}".format(crop_mask.shape))
    crop_image = tf.image.random_crop(crop_image, size = [IMG_SIZE, IMG_SIZE, 3], seed=SEED)
    crop_mask = tf.image.random_crop(crop_mask, size = [IMG_SIZE, IMG_SIZE, 1], seed=SEED)
    print("Shape image after crop: {}".format(crop_image.shape))
    print("Shape mask after crop: {}".format(crop_mask.shape))    
    return crop_image, crop_mask


# @tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE), method='area')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method='area')

    input_image = input_image / 255
    input_mask = tf.image.rgb_to_grayscale(input_mask)
    input_mask = tf.floor(input_mask / 255 + 0.5)

    return input_image, input_mask


train_imgs = glob(Train_Images_Path + "*.jpg")
val_imgs = glob(Val_Images_Path + "*.jpg")
TRAIN_LENGTH = len(train_imgs)
VAL_LENGTH = len(val_imgs)
print('train lenght: ', TRAIN_LENGTH)
print('val lenght: ', VAL_LENGTH)


BUFFER_SIZE = BATCH_SIZE
STEPS_PER_EPOCH = (TRAIN_LENGTH // BATCH_SIZE)
print('steps per epoch: ', STEPS_PER_EPOCH)

train = dataset['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)


# print(tf.data.Dataset.cardinality(train).numpy())        # how big is the dataset
# print(tf.data.Dataset.cardinality(test).numpy())


train_dataset = train.cache() #.shuffle(buffer_size=TRAIN_LENGTH, seed=SEED, reshuffle_each_iteration=True)
#train_dataset = train_dataset.map(train_random_crop, num_parallel_calls=AUTOTUNE)                                                       #  apply random_crop | if disabled adjust image resize in load_image_train
train_dataset = train_dataset.batch(BATCH_SIZE).repeat() #
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)




test = dataset['test'].map(load_image_test)
test_dataset = test.batch(BATCH_SIZE)
test_dataset = test_dataset.cache().repeat()


# Visualizing the Loaded Dataset

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        print("display sample shape: {}".format(display_list[i].shape))
        # print("Type: {}".format(display_list[i].dtype))
        # print("Mean: {}".format(tf.math.reduce_mean(display_list[i])))
        
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Test dataset images and masks

for image, mask in train_dataset.take(1):
    sample_image, sample_mask = image[0], mask[0]
    print("Shape image[0] dataset.take(1): {}".format(sample_image[0].shape))
    print("Shape mask[0] dataset.take(1): {}".format(sample_mask[0].shape))
    break     


SIZE = IMG_SIZE

def pyramid_pooling(input_tensor, sub_region_sizes):
    """This class implements the Pyramid Pooling Module
    WARNING: This function uses eager execution, so it only works with
        Tensorflow 2.0 backend.
    Args:
        input_tensor: Tensor with shape: (batch, rows, cols, channels)
        sub_region_sizes: A list containing the size of each region for the
            sub-region average pooling. The default value is [1, 2, 3, 6]
    Returns:
        output_tensor: Tensor with shape: (batch, rows, cols, channels * 2)
    """
    _, input_height, input_width, input_channels = input_tensor.shape
    feature_maps = [input_tensor]
    for i in sub_region_sizes:
        curr_feature_map = keras.layers.AveragePooling2D(
            pool_size=(input_height // i, input_width // i),
            strides=(input_height // i, input_width // i))(input_tensor)
        curr_feature_map = keras.layers.Conv2D(
            filters=int(input_channels) // len(sub_region_sizes),
            kernel_size=3,
            padding='same')(curr_feature_map)
        curr_feature_map = keras.layers.Lambda(
            lambda x: tf.image.resize(
                x, (input_height, input_width)))(curr_feature_map)
        feature_maps.append(curr_feature_map)

    output_tensor = keras.layers.Concatenate(axis=-1)(feature_maps)

    output_tensor = keras.layers.Conv2D(
        filters=128, kernel_size=3, strides=1, padding="same")(
        output_tensor)
    output_tensor = keras.layers.BatchNormalization()(output_tensor)
    output_tensor = keras.layers.Activation("relu")(output_tensor)
    return output_tensor

def bottleneck(input_tensor, filters, strides, expansion_factor):
    """Implementing Bottleneck.
    This class implements the bottleneck module for Fast-SCNN.
    Layer structure:
        ----------------------------------------------------------------
        |  Input shape   |  Block  |  Kernel | Stride |  Output shape  |
        |                |         |   size  |        |                |
        |----------------|---------|---------|--------|----------------|
        |   h * w * c    |  Conv2D |    1    |    1   |   h * w * tc   |
        |----------------|---------|---------|--------|----------------|
        |   h * w * tc   |  DWConv |    3    |    s   | h/s * w/s * tc |
        |----------------|---------|---------|--------|----------------|
        | h/s * w/s * tc |  Conv2D |    1    |    1   | h/s * w/s * c` |
        |--------------------------------------------------------------|
        Designations:
            h: input height
            w: input width
            c: number of input channels
            t: expansion factor
            c`: number of output channels
            DWConv: depthwise convolution
    Args:
        input_tensor: Tensor with shape: (batch, rows, cols, channels)
        filters: Output filters
        strides: Stride used in depthwise convolution layer
        expansion_factor: hyperparameter
    Returns:
        output_tensor: Tensor with shape: (batch, rows // stride,
            cols // stride, new_channels)
    """
    _, input_height, input_width, input_channels = input_tensor.shape
    tensor = keras.layers.Conv2D(
        filters=input_channels * expansion_factor,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu")(input_tensor)
    tensor = keras.layers.BatchNormalization()(tensor)
    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.DepthwiseConv2D(kernel_size=3,
                                          strides=strides,
                                          padding="same")(tensor)
    tensor = keras.layers.BatchNormalization()(tensor)
    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.Conv2D(filters=filters,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same")(tensor)
    tensor = keras.layers.BatchNormalization()(tensor)
    output_tensor = keras.layers.Activation('relu')(tensor)
    return output_tensor

def create_fast_scnn(num_classes, input_shape=[None, None, 3], sub_region_sizes=[1, 2, 3, 6], expansion_factor=6):
    """This function creates a Fast-SCNN neural network model using
    the Keras functional API.
    Args:
        num_classes: Number of classes
        input_shape: A list containing information about the size of the image.
            List structure: (rows, cols, channels). Dimensions can also be
            None if they can be of any size.
        expansion_factor: Hyperparameter in the bottleneck layer
        sub_region_sizes: A list containing the sizes of subregions for
            average pool by region in the pyramidal pool module
    Returns:
        model: uncompiled Keras model
    """

    # Sub-models for every Fast-SCNN block

    input_tensor = keras.layers.Input(input_shape)

    learning_to_down_sample = keras.layers.Conv2D(
        32, 3, 2, padding="same")(input_tensor)
    learning_to_down_sample = keras.layers.BatchNormalization()(
        learning_to_down_sample)
    learning_to_down_sample = keras.layers.Activation("relu")(
        learning_to_down_sample)

    learning_to_down_sample = keras.layers.SeparableConv2D(
        48, 3, 2, padding="same")(learning_to_down_sample)
    learning_to_down_sample = keras.layers.BatchNormalization()(
        learning_to_down_sample)
    learning_to_down_sample = keras.layers.Activation("relu")(
        learning_to_down_sample)

    learning_to_down_sample = keras.layers.SeparableConv2D(
        64, 3, 2, padding="same")(learning_to_down_sample)
    learning_to_down_sample = keras.layers.BatchNormalization()(
        learning_to_down_sample)
    learning_to_down_sample = keras.layers.Activation("relu")(
        learning_to_down_sample)

    skip_connection = learning_to_down_sample

    # Global feature extractor

    global_feature_extractor = bottleneck(learning_to_down_sample,
                                          64, 2, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          64, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          64, 1, expansion_factor)

    global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 2, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)

    global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
    global_feature_extractor = pyramid_pooling(global_feature_extractor,
                                               sub_region_sizes)

    # Feature fusion

    feature_fusion_main_branch = keras.layers.UpSampling2D((4, 4))(
        global_feature_extractor)

    feature_fusion_main_branch = keras.layers.DepthwiseConv2D(
        3, padding="same")(feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.BatchNormalization()(
        feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.Activation("relu")(
        feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.Conv2D(
        128, 1, 1, padding="same")(feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.BatchNormalization()(
        feature_fusion_main_branch)

    feature_fusion_skip_connection = keras.layers.Conv2D(
        128, 1, 1, padding="same")(skip_connection)
    feature_fusion_skip_connection = keras.layers.BatchNormalization()(
        feature_fusion_skip_connection)

    feature_fusion = feature_fusion_main_branch + feature_fusion_skip_connection

    # Classifier

    classifier = keras.layers.SeparableConv2D(128, 3, 1, padding="same")(
        feature_fusion)
    classifier = keras.layers.BatchNormalization()(classifier)
    classifier = keras.layers.Activation("relu")(classifier)

    classifier = keras.layers.SeparableConv2D(128, 3, 1, padding="same")(
        classifier)
    classifier = keras.layers.BatchNormalization()(classifier)
    classifier = keras.layers.Activation("relu")(classifier)

    classifier = keras.layers.Conv2D(num_classes, 3, 1, padding="same")(
        classifier)
    classifier = keras.layers.BatchNormalization()(classifier)
    classifier = keras.layers.Activation("relu")(classifier)

    output_tensor = keras.layers.UpSampling2D((8, 8))(classifier)
    output_tensor = keras.layers.Softmax()(output_tensor)

    model = keras.models.Model(input_tensor, output_tensor)
    return model

model = create_fast_scnn(2, input_shape=[SIZE, SIZE, 3] )

class MaskMeanIoU(tf.keras.metrics.MeanIoU):
    #                                                                                                    Mean Intersection over Union
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


def tversky(y_true, y_pred):
    alpha = 0.7
    smooth = 1.0
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def dsc(y_true, y_pred, eps=1e-6):
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)
    y_pred_class_f = tf.keras.backend.flatten(y_pred) # the dice loss implementation
    y_true_f = tf.keras.backend.flatten(y_true)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_class_f)
    answer = (2. * intersection + eps) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_class_f) + eps)
    return answer


optimizer_Adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', MaskMeanIoU(name='iou', num_classes=OUTPUT_CHANNELS), dsc, tversky], #
              )  #run_eagerly=True                                                                                                                     # run eager

model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)                                                                                                # plot model


def create_mask(pred_mask):                                                                                                                         # create mask
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display_sample([image[0], mask[0], create_mask(pred_mask)])
  else:
    display_sample([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

#                                                                                                                                          load weights from last save
# if os.path.exists("./Weights/U-net_128_16bit_model_initializer.h5"): 
#     model.load_weights("./Weights/U-net_128_16bit_model_initializer.h5")
#     print("Model loded - OK")

# show_predictions()
# model.predict(sample_image[tf.newaxis, ...])

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
  if epoch < 6:
    return 0.0005
  else:
    return 0.0001 #* tf.math.exp(0.1 * (10 - epoch))

LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#  - TensorBoard
data_folder = Path("c:/TFlogs/fit/")
log_dir=data_folder / datetime.datetime.now().strftime("%m%d-%H%M%S")  #folder for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True, write_graph=True) #, profile_batch='50,500', histogram_freq=1, write_graph=True

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        # show_predictions()
        show_predictions(train_dataset, 2)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
        model.save_weights("./Weights/U-net_512_v2_model.h5")



VALIDATION_STEPS = VAL_LENGTH // BATCH_SIZE

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tensorboard_callback, LRS])  #LRS, 



show_predictions(train_dataset, 3)
show_predictions(test_dataset, 3)
