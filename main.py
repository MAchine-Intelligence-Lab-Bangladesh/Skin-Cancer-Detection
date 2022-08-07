import numpy as np

from data_preparation import data_preparation


def ISIC_2019_data_preprocess():
    # df = pd.read_csv("data/ISIC_2019_3_class/raw_data/ISIC_2019_Training_GroundTruth.csv")
    # mel_list = df[df['MEL'] == 1.0]['image'].tolist()
    # nv_list = df[df['NV'] == 1.0]['image'].tolist()
    # bcc_list = df[df['BCC'] == 1.0]['image'].tolist()
    #
    a = data_preparation()
    # a.data_separation(data_list=mel_list, source="data/ISIC_2019_3_class/raw_data/ISIC_2019_Training_Input/",
    #                   destination="data/ISIC_2019_3_class/processed_data/Melanoma/",
    #                   extension="MEL")
    # print("Completed melanoma")
    # a.data_separation(data_list=nv_list, source="data/ISIC_2019_3_class/raw_data/ISIC_2019_Training_Input/",
    #                   destination="data/ISIC_2019_3_class/processed_data/Nevus/",
    #                   extension="NV")
    # print("completed nevus")
    # a.data_separation(data_list=bcc_list, source="data/ISIC_2019_3_class/raw_data/ISIC_2019_Training_Input/",
    #                   destination="data/ISIC_2019_3_class/processed_data/Basal_cell_carcinoma/", extension="BCC")
    # print("completed bcc")
    # print(f"melanoma: {len(mel_list)}")
    # print(f"nv: {len(nv_list)}")
    # print(f"bcc: {len(bcc_list)}")

    a.data_extraction()


def main():
    print("enter")
    # ISIC_2019_data_preprocess()
    feats = np.load("data/ISIC_2019_3_class/processed_data/feats_train.npy")
    labels = np.load("data/ISIC_2019_3_class/processed_data/labels_train.npy")
    print("loading done")

    a = data_preparation()
    feats, labels = a.randomize(feats, labels)
    print("randomization done")

    # splitting 80:20 ratio i.e., 80% for training and 20% for testing purpose
    (x_train, x_test) = feats[int(0.2 * len(feats)):], feats[:int(0.2 * len(feats))]
    (y_train, y_test) = labels[int(0.2 * len(feats)):], labels[:int(0.2 * len(feats))]
    print("splitting done")

    x_train, x_test, y_train, y_test = a.normalize(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print("normalization done")

    train_aug = a.image_augmentation()
    print("augmentation done")


if __name__ == '__main__':
    main()
