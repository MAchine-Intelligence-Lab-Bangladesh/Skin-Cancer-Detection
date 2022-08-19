import numpy as np
import pandas as pd

from experiment_model.Model_MobileNetV2 import mobilenetV2
from prepare_dataset.isic_2019_data_prep import data_preparation


def isic_2020_data_preprocess():
    raw_dir_path = "data/isic_2020/raw_data/isic_2020_train/"
    processed_dir_path = "data/isic_2020/processed_data/"
    df = pd.read_csv("data/isic_2020/raw_data/ISIC_2020_Training_GroundTruth.csv")
    benign_list = df[df['type'] == "benign"]['image_name'].tolist()
    malignant_list = df[df['type'] == "malignant"]['image_name'].tolist()

    a = data_preparation()
    a.data_separation(data_list=benign_list, source=raw_dir_path,
                      destination=processed_dir_path,
                      extension="ben")
    a.data_separation(data_list=malignant_list, source=raw_dir_path,
                      destination=processed_dir_path,
                      extension="mal")
    print(f"melanoma: {len(benign_list)}")
    print(f"nv: {len(malignant_list)}")

    a.data_extraction(dir_path=processed_dir_path, item_list=["benign", "malignant"])


def isic_2020_train_model():
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
