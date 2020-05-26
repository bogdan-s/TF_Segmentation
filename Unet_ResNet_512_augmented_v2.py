#lowering the model so it doesn t overfit

import os
import sys
from pathlib import Path
import datetime
# from IPython.display import clear_output
# import IPython.display as display
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
IMG_SIZE_Before_Crop = 530 #150 for 128 final image
IMG_SIZE = 512
BATCH_SIZE = 6
OUTPUT_CHANNELS = 2
EPOCHS = 60
away_from_computer = True  # to show or not predictions between batches
save_model_for_inference = False # to save or not the model for inference
SEED = 19

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


train_dataset = train.cache().shuffle(buffer_size=TRAIN_LENGTH, seed=SEED, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(train_random_crop, num_parallel_calls=AUTOTUNE)                                                       #  apply random_crop | if disabled adjust image resize in load_image_train
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

def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c


def ResUNet():
    f1 = [32, 64, 128, 256, 512]
    f = [_/4 for _ in f1]
    inputs = keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    
    ## Bridge
    b0 = residual_block(e4, f[4], strides=2)

    ## Decoder
    u1 = upsample_concat_block(b0, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(2, (1,1), padding="same", activation="softmax")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = ResUNet()

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

#   #######                                                                                                                                 load weights from last save
if os.path.exists("./Weights/U-net_ResNet_512_v1_4_model.h5"): 
    model.load_weights("./Weights/U-net_ResNet_512_v1_4_model.h5")
    print("Model loded - OK")

# show_predictions()
# model.predict(sample_image[tf.newaxis, ...])

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
#   if epoch < 6:
#     return 0.0005
#   else:
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
        # show_predictions(train_dataset, 1)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
        model.save("./Weights/U-net_ResNet_512_v1_4_model")



VALIDATION_STEPS = VAL_LENGTH // BATCH_SIZE

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tensorboard_callback, LRS])  #LRS, 



show_predictions(train_dataset, 3)
show_predictions(test_dataset, 3)

