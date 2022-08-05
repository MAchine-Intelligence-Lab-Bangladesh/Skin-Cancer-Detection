import os
import shutil

import pandas as pd


def data_separation(data_list, source, destination, extension=""):
    allfiles = os.listdir(source)

    for f in allfiles:
        m = f.split('.')[0]
        if m in data_list:
            file = extension + "_" + str(f)
            print(file)
            shutil.copy(source + f, destination + file)


def read_data():
    df = pd.read_csv("data/ISIC_2019_Training_GroundTruth.csv")
    mel_list = df[df['MEL'] == 1.0]['image'].tolist()
    nv_list = df[df['NV'] == 1.0]['image'].tolist()
    bcc_list = df[df['BCC'] == 1.0]['image'].tolist()
    data_separation(data_list=mel_list, source="data/ISIC_2019_Training_Input/", destination="data/Melanoma/",
                    extension="MEL")
    data_separation(data_list=nv_list, source="data/ISIC_2019_Training_Input/", destination="data/Nevus/",
                    extension="NV")
    data_separation(data_list=bcc_list, source="data/ISIC_2019_Training_Input/",
                    destination="data/Basal_cell_carcinoma/", extension="BCC")
    print(f"melanoma: {len(mel_list)}")
    print(f"nv: {len(nv_list)}")
    print(f"bcc: {len(bcc_list)}")


if __name__ == '__main__':
    read_data()
