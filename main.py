from prepare_dataset.isic_2020_data_prep import data_preparation


def main():
    # isic_2020_data_preprocess()
    # isic_2020_train_model()
    a = data_preparation()
    a.data_resize(class_name="benign", img_width=224, img_height=224)


if __name__ == '__main__':
    main()
