import os
import shutil

import cv2
import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.utils.np_utils import to_categorical


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

    def data_extraction(self):
        print("appended")
        self.extract("Basal_cell_carcinoma")
        self.extract("Nevus")
        self.extract("Melanoma")
        feats = np.array(self.data)
        labels = np.array(self.labels)
        print("np saving...")
        np.save("data/ISIC_2019_3_class/processed_data/feats_train", feats)
        np.save("data/ISIC_2019_3_class/processed_data/labels_train", labels)

        print("extraction done")

    def extract(self, dir_path):
        path = os.listdir("data/ISIC_2019_3_class/processed_data/" + dir_path + "/")
        for a in path:
            image = cv2.imread("data/ISIC_2019_3_class/processed_data/" + dir_path + "/" + a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((224, 224))
            self.data.append(np.array(size_image))
            if dir_path == "Basal_cell_carcinoma":
                self.labels.append(0)
            elif dir_path == "Nevus":
                self.labels.append(1)
            else:
                self.labels.append(2)
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
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        train_len = len(x_train)
        test_len = len(x_test)

        y_train = to_categorical(y_train, 3)
        y_test = to_categorical(y_test, 3)
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
