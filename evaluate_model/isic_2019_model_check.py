import numpy as np
import pandas as pd

from experiment_model.Model_MobileNetV2 import mobilenetV2
from prepare_dataset.isic_2019_data_prep import data_preparation


def isic_2019_data_preprocess():
    df = pd.read_csv("data/isic_2019/raw_data/ISIC_2019_Training_GroundTruth.csv")
    mel_list = df[df['MEL'] == 1.0]['image'].tolist()
    nv_list = df[df['NV'] == 1.0]['image'].tolist()
    bcc_list = df[df['BCC'] == 1.0]['image'].tolist()
    #
    a = data_preparation()
    a.data_separation(data_list=mel_list, source="data/isic_2019/raw_data/ISIC_2019_Training_Input/",
                      destination="data/isic_2019/processed_data/Melanoma/",
                      extension="MEL")
    a.data_separation(data_list=nv_list, source="data/isic_2019/raw_data/ISIC_2019_Training_Input/",
                      destination="data/isic_2019/processed_data/Nevus/",
                      extension="NV")
    a.data_separation(data_list=bcc_list, source="data/isic_2019/raw_data/ISIC_2019_Training_Input/",
                      destination="data/isic_2019/processed_data/Basal_cell_carcinoma/", extension="BCC")
    print(f"melanoma: {len(mel_list)}")
    print(f"nv: {len(nv_list)}")
    print(f"bcc: {len(bcc_list)}")

    a.data_extraction()


def isic_2019_train_model():
    print("enter")
    # isic_2019_data_preprocess()
    feats = np.load("data/isic_2019/processed_data/feats_train.npy")
    labels = np.load("data/isic_2019/processed_data/labels_train.npy")
    print("loading done")

    a = data_preparation()
    feats, labels = a.randomize(feats, labels)
    print("randomization done")

    # splitting 80:20 ratio i.e., 80% for training and 20% for testing purpose
    (x_train, x_test) = feats[int(0.2 * len(feats)):], feats[:int(0.2 * len(feats))]
    (y_train, y_test) = labels[int(0.2 * len(feats)):], labels[:int(0.2 * len(feats))]
    print("splitting done")

    x_train, x_test, y_train, y_test = a.normalize(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print("categorization done")

    train_aug = a.image_augmentation()
    print("augmentation done")

    mobilenetV2(train_aug=train_aug, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print("model saved.")
