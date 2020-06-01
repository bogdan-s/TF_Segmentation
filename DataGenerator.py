import tensorflow as tf
import math


class CustomDataGenerator(tf.data.Dataset):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_path(cls, train_images_path, image_size = 512, image_crop = True, shuffle=True, seed=2020, batch_size = 16):
        """
        Inputs:
        train_images_path: string 
            path to train images - the test images path will be generated 
        image_size: int
            final image size that will be feed in the model
        image_crop: bool
            whether the train images will be random croped or not
        seed: int
            2020 seems quite a good seed for random things :)
        """
        # tf.random.set_seed(seed)

        #get file lists
        train = cls.list_files(train_images_path + "*.jpg", shuffle=False)
        
        test_images_path = tf.strings.regex_replace(train_images_path, "Train", "Validation")
        test = cls.list_files(test_images_path + "*.jpg", shuffle=False)
        
        train_length = tf.data.experimental.cardinality(train).numpy()
        test_length = tf.data.experimental.cardinality(test).numpy()

        #generate generators of dictionaries of images and masks - dtype - uint8
        train = train.map(cls.parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = test.map(cls.parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        #group them in a dictionary so that it's easier to apply the same transformations on both the image and mask
        dataset = {"train": train, "test": test}

        #function that prepreocesses the train images
        def load_image_train(dataset) -> tuple:
        
            image_size_before_crop = image_size
            
            #set bigger size before crop if crop is wanted
            if image_crop:
                image_size_before_crop = math.ceil(image_size * 1.05)

            # resize the images
            image = tf.image.resize(dataset["image"], (image_size_before_crop, image_size_before_crop), method="area")
            mask = tf.image.resize(dataset["mask"], (image_size_before_crop, image_size_before_crop), method="area")



            #flip them
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)

            #normalize the images and masks
            image = image / 255
            mask = tf.image.rgb_to_grayscale(mask)
            
            # this works only for binary black@white masks
            mask = tf.floor(mask / 255 + 0.5)

            return image, mask


        def crop_image_train(image_to_crop, mask_to_crop) -> tuple:
            """
            because the random crop had inconsistencies (different seeds) between the image and the mask 
            i combined them in one tensor that i random crop and after that i split them back
            """
            concatenated = tf.keras.backend.concatenate((image_to_crop, mask_to_crop), axis=-1)
            concatenated = tf.image.random_crop(concatenated, size=[image_size, image_size, 4])
            image = concatenated[:,:,:-1]
            mask = concatenated[:,:,-1:]
            mask = tf.cast(mask, tf.int64)
            return image, mask

        #function that prepreocesses the test images

        def load_image_test(dataset) -> tuple:
            #resize the images
            image = tf.image.resize(dataset["image"], (image_size,image_size), method="area")
            mask = tf.image.resize(dataset["mask"], (image_size,image_size), method="area")

            # normalize them
            image = image / 255
            mask = tf.image.rgb_to_grayscale(mask)

            # this works only for binary black@white masks
            mask = tf.floor(mask / 255 + 0.5)

            return image, mask

        train = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle:
            train = train.cache().shuffle(buffer_size=train_length, reshuffle_each_iteration=True)

        train = train.map(crop_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train = train.batch(batch_size).repeat()
        train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        test = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = test.batch(batch_size)
        test = test.cache().repeat()

        return (train, test, train_length, test_length)

    @staticmethod
    def parse_image(img_path):
        """
        Input: tf string
        ------------------------------------
        read Train images
        generate the paths for masks
        read Train masks
        ------------------------------------
        Output: dict with image & mask pairs
        """
        image = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(image, channels=3)
        # mask = image
        mask_path = tf.strings.regex_replace(img_path, "Images", "New_Masks")
        mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_seg.png")
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask, channels=0, dtype=tf.dtypes.uint8)

        return {"image": image, "mask": mask}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # dataset location
    train_images_path = "D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/"


    train_dataset, test_dataset, train_length, test_length = CustomDataGenerator.from_path(
                                                                train_images_path, 
                                                                seed=2020, 
                                                                shuffle=False,
                                                                batch_size=256,
                                                                image_crop=True) #, image_size=512, image_crop=True, seed=2020, batch_size=64
    
    print(f"Train length: {train_length}")
    print(f"Test length: {test_length}")

    def display_sample_overlap(display_list):
        """Show side-by-side an input image,
        the ground truth and the prediction.
        """
        plt.figure(figsize=(18, 18))

        title = ['Input Image', 'True Mask', 'Predicted Mask']
        a = 0
        for i in range(len(display_list)):
            print(display_list[i].shape)
            a += display_list[i] / (i + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(a))
        plt.show()
    print(train_dataset)
    i = 0
    for (batch, (images, masks)) in enumerate(train_dataset):
        print(batch)
        display_sample_overlap([images[0], masks[0]])
        # print(images)
        # print(masks)
        if i == 1:
            break



