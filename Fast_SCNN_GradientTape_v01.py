

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
IMG_SIZE_Before_Crop = 530 #150 for 128 final image
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
    
    mask_path = tf.strings.regex_replace(img_path, "Images", "Edges")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_edg.png")
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


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f1 = [32, 64, 128, 256, 512]
    f = [_/4 for _ in f1]
    inputs = keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(2, (1,1), padding="same", activation="softmax")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

#Fast scnn from https://github.com/rudolfsteiner/fast_scnn/blob/master/Fast_SCNN.py

def down_sample(input_layer):
    
    ds_layer = tf.keras.layers.Conv2D(32, (3,3), padding='same', strides = (2,2))(input_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    ds_layer = tf.keras.layers.SeparableConv2D(48, (3,3), padding='same', strides = (2,2))(ds_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    ds_layer = tf.keras.layers.SeparableConv2D(64, (3,3), padding='same', strides = (2,2))(ds_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    return ds_layer
    

def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    #x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))
    x = tf.keras.layers.Conv2D(tchannel, (1,1), padding='same', strides = (1,1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #x = #conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)
    
    x = tf.keras.layers.Conv2D(filters, (1,1), padding='same', strides = (1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)


    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

        return x

def global_feature_extractor(lds_layer):
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8], gfe_layer.shape[1], gfe_layer.shape[2])
    print("gfe_layer.shape:", gfe_layer.shape)
    
    return gfe_layer

def pyramid_pooling_block(input_tensor, bin_sizes, w, h):
    print(w, h)
    concat_list = [input_tensor]
    #w = 16 # 64
    #h = 16 #32

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), 
                                             strides=(w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)
        print("x in paramid.shape", x.shape)

    concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)

def feature_fusion(lds_layer, gfe_layer):
    ff_layer1 = tf.keras.layers.Conv2D(128, (1,1), padding='same', strides = (1,1))(lds_layer)
    ff_layer1 = tf.keras.layers.BatchNormalization()(ff_layer1)
    #ff_layer1 = tf.keras.activations.relu(ff_layer1)
    print("ff_layer1.shape", ff_layer1.shape)
    
    #ss = conv_block(gfe_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)
    #print(ss.shape, ff_layer1.shape)
    

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    print("ff_layer2.shape", ff_layer2.shape)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    
    print("ff_layer2.shape", ff_layer2.shape)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)
    
    print("ff_layer2.shape", ff_layer2.shape)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)
    
    print("ff_final.shape", ff_final.shape)
    
    return ff_final

def classifier_layer(ff_final, num_classes):
    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', 
                                                 strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)

    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', 
                                                 strides = (1, 1), name = 'DSConv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)
    #change 19 to 20
    #classifier = conv_block(classifier, 'conv', 20, (1, 1), strides=(1, 1), padding='same', relu=True)

    classifier = tf.keras.layers.Conv2D(num_classes, (1,1), padding='same', strides = (1,1))(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)
    
    classifier = tf.keras.layers.Dropout(0.3)(classifier)
    print("classifier before upsampling:", classifier.shape)

    classifier = tf.keras.activations.softmax(classifier)
    
    return classifier

def get_fast_scnn(w, h, num_classes):
    """
    input image: (w, h)
    """
    
    input_layer = tf.keras.layers.Input(shape=(w, h, 3), name = 'input_layer')
    ds_layer = down_sample(input_layer)
    gfe_layer = global_feature_extractor(ds_layer)
    ff_final = feature_fusion(ds_layer, gfe_layer)
    classifier = classifier_layer(ff_final, num_classes)
    
    fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
    
    return fast_scnn

model = UNet()

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

loss_object = dsc
#loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
loss_history = []



# model.summary()

# model.load_weights("./Weights/U-net_128_16bit_model.h5")

def create_mask(pred_mask):                                                                                                                         # create mask
    # print("Predicted mask: {}".format(pred_mask))
    # print("Predicted mask: {}".format(pred_mask.shape))
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # print("Predicted mask after argmax: {}".format(pred_mask.shape))
    pred_mask = pred_mask[..., tf.newaxis]
    # print("Predicted mask after added new axis: {}".format(pred_mask.shape))
    # print("Predicted mask[0]: {}".format(pred_mask[0]))
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display_sample([image[0], mask[0], create_mask(pred_mask)])
  else:
    display_sample([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions(train_dataset, 1)

def visualize_trainable_vars(model):
    """ function to visualize the trainable variables values. takes a complied model as argument"""
    trainable_vars_init = [tf.keras.backend.mean(_).numpy() for _ in model.trainable_variables]
    trainable_vars_names = [_.name for _ in model.trainable_variables]
    print('Mean values per trainable_vars_init {}'.format(trainable_vars_init))
    print('Trainable var names {}'.format(trainable_vars_names))
    x_pos = [i for i, _ in enumerate(trainable_vars_names)]
    bars = plt.barh(x_pos, trainable_vars_init)
    plt.yticks(x_pos, zip(trainable_vars_names, trainable_vars_init), rotation=0)
    plt.show()

visualize_trainable_vars(model)

def train_step(images, masks):
    with tf.GradientTape() as tape:
        logits = model(images)
        print('Logits: {}'.format(logits))
        print('Logits shape: {}'.format(logits.shape))
        # print('masks: {}'.format(masks))
        loss_value = loss_object(masks, logits)
        print('loss value {}'.format(loss_value))
    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, model.trainable_variables)
    grad_list = [tf.keras.backend.mean(_).numpy() for _ in grads]
    print('Mean values per gradient {}'.format(grad_list))
    plt.plot(grad_list)
    plt.show()
    optimizer_Adam.apply_gradients(zip(grads, model.trainable_variables))
    # print(model.predict(sample_image[tf.newaxis, ...]))
    # show_predictions(train_dataset, 15)

def train(epochs):
    for epoch in range(epochs):
        no_of_batches = 0
        for (batch, (images, masks)) in enumerate(train_dataset):
            no_of_batches += 1
            if no_of_batches < 3:
                print('Batch no {} of epoch {}'.format(batch, epoch))
                # print('Images shape: {}'.format(images.shape))
                # print('Masks shape: {}'.format(masks.shape))
                train_step(images, masks)
            else: 
                break
        print ('Epoch {} finished'.format(epoch))
        # visualize_trainable_vars(model)
        # show_predictions(train_dataset, 1)

train(epochs = 2)



show_predictions()