import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical


class data_preparation:
    def __init__(self):
        self.data = []
        self.labels = []

    def data_separation(self, data_list, source, destination, extension=""):
        allfiles = os.listdir(source)

        for f in allfiles:
            m = f.split('.')[0]
            if m in data_list:
                file = extension + "_" + str(f)
                print(file)
                shutil.copy(source + f, destination + file)

    def data_resize(self, class_name, img_height, img_width):
        i = 0
        root_path = "data/isic_2020/resized_data"
        resized_path = root_path + "/" + str(img_height) + "*" + str(img_width)
        if not os.path.exists(resized_path):
            os.mkdir(resized_path)
        if not os.path.exists(resized_path + "/" + class_name):
            os.mkdir(resized_path + "/" + class_name)
        else:
            print("class and resolution folder are created. ")
        for image in glob.glob("data/isic_2020/processed_data/" + class_name + "/*.jpg"):
            image = cv2.imread(image)
            imgResized = cv2.resize(image, (img_width, img_height))
            cv2.imwrite(filename=resized_path + "/" + class_name + "image%i.jpg" % i, img=imgResized)
            print(i)

    def data_extraction(self, dir_path, item_list):
        for item in item_list:
            print(f"{item} processing...")
            self.extract(dir_path=dir_path, folder_name=item)
            print(f"{item} done")
        feats = np.array(self.data)
        labels = np.array(self.labels)
        print("np saving...")
        np.save(dir_path + "/feats_train", feats)
        np.save(dir_path + "/labels_train", labels)

        print("extraction done")

    def extract(self, dir_path, folder_name):
        path = os.listdir(dir_path + "/" + folder_name + "/")
        for a in path:
            image = cv2.imread(dir_path + "/" + folder_name + "/" + a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((224, 224))
            self.data.append(np.array(size_image))
            if folder_name == "benign":
                self.labels.append(0)
            else:
                self.labels.append(1)
            print(a)

    def randomize(self, features, labels):
        s = np.arange(features.shape[0])
        np.random.shuffle(s)
        feats = features[s]
        labels = labels[s]
        num_classes = len(np.unique(labels))
        print(num_classes)
        return feats, labels

    def normalize(self, x_train, x_test, y_train, y_test):
        # x_train = x_train.astype('float32')  
        # x_test = x_test.astype('float32')
        # train_len = len(x_train)
        # test_len = len(x_test)

        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)
        print("normalization done")
        return x_train, x_test, y_train, y_test

    def image_augmentation(self):
        trainAug = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)
        print("augmentation done")
        return trainAug
