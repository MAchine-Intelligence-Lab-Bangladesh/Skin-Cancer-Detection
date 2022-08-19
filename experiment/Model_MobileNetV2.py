from tensorflow.keras import Model, layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model


# https://colab.research.google.com/drive/17s15JY0qg26yawX1htPJGl8yJGvOSXV6#scrollTo=nfiDdL3uJmOg
def mobilenetV2(train_aug, x_train, x_test, y_train, y_test):
    conv_base = MobileNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet')

    for layer in conv_base.layers:
        layer.trainable = True

    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    predictions = layers.Dense(3, activation='softmax')(x)
    model = Model(conv_base.input, predictions)

    callbacks = [ModelCheckpoint('.mdl_wts.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='min',
                                   min_lr=0.00000000001)]

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    BS = 64
    print("training head...")
    model.fit(
        train_aug.flow(x_train, y_train, batch_size=BS),
        steps_per_epoch=len(x_train) // BS,
        validation_data=(x_test, y_test),
        validation_steps=len(x_test) // BS,
        epochs=30, callbacks=callbacks)

    model = load_model('.mdl_wts.hdf5')
    model.save('saved_model/mobilenet_v1.h5')
