import pandas as pd

from data_preparation import data_preparation


def ISIC_2019_data_preprocess():
    df = pd.read_csv("data/ISIC_2019_Training_GroundTruth.csv")
    mel_list = df[df['MEL'] == 1.0]['image'].tolist()
    nv_list = df[df['NV'] == 1.0]['image'].tolist()
    bcc_list = df[df['BCC'] == 1.0]['image'].tolist()

    a = data_preparation()
    a.data_separation(data_list=mel_list, source="data/ISIC_2019_Training_Input/", destination="data/Melanoma/train/",
                      extension="MEL")
    a.data_separation(data_list=nv_list, source="data/ISIC_2019_Training_Input/", destination="data/Nevus/train/",
                      extension="NV")
    a.data_separation(data_list=bcc_list, source="data/ISIC_2019_Training_Input/",
                      destination="data/Basal_cell_carcinoma/train/", extension="BCC")
    print(f"melanoma: {len(mel_list)}")
    print(f"nv: {len(nv_list)}")
    print(f"bcc: {len(bcc_list)}")

    a.data_extraction()


def main():
    print("enter")
    # feats = np.load("data/feats_train.npy")
    # labels = np.load("data/labels_train.npy")
    # print("loading done")
    #
    # a = data_preparation()
    # feats, labels = a.randomize(feats, labels)
    # print("randomization done")
    #
    # # splitting 80:20 ratio i.e., 80% for training and 20% for testing purpose
    # (x_train, x_test) = feats[int(0.2 * len(feats)):], feats[:int(0.2 * len(feats))]
    # (y_train, y_test) = labels[int(0.2 * len(feats)):], labels[:int(0.2 * len(feats))]
    # print("spplitting done")
    #
    # x_train, x_test, y_train, y_test = a.randomize(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    # print("randomize done")
    #
    # train_aug = a.image_augmentation()


if __name__ == '__main__':
    main()
