

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
from DataGenerator import CustomDataGenerator



# # Device Compute Precision - mixed precision
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__, end='\n\n')
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
# param

IMG_SIZE = 512
BATCH_SIZE = 6
OUTPUT_CHANNELS = 2
EPOCHS = 60
away_from_computer = True  # to show or not predictions between batches
save_model_for_inference = False # to save or not the model for inference
SEED = 15

# dataset location
Train_Images_Path = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/"

train_dataset, test_dataset, train_length, test_length = CustomDataGenerator.from_path(
                                                                                        Train_Images_Path,
                                                                                        image_size=IMG_SIZE, 
                                                                                        seed=SEED, 
                                                                                        shuffle=True,
                                                                                        batch_size=BATCH_SIZE,
                                                                                        image_crop=True
                                                                                        )



STEPS_PER_EPOCH = train_length // BATCH_SIZE
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
        # print("display sample shape: {}".format(display_list[i].shape))
        # print("Type: {}".format(display_list[i].dtype))
        # print("Mean: {}".format(tf.math.reduce_mean(display_list[i])))
        
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Test dataset images and masks

for image, mask in train_dataset.take(1):
    sample_image, sample_mask = image[0], mask[0]
    # print("Shape image[0] dataset.take(1): {}".format(sample_image.shape))
    # print("Shape mask[0] dataset.take(1): {}".format(sample_mask.shape))
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

show_predictions()
# print(model.predict(sample_image[tf.newaxis, ...]))

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
        show_predictions(train_dataset, 1)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
        model.save_weights("./Weights/U-net_512_v3_model.h5")



VALIDATION_STEPS = test_length // BATCH_SIZE

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tensorboard_callback, LRS])  #LRS, 



show_predictions(train_dataset, 3)
show_predictions(test_dataset, 3)

