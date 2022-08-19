import itertools

import matplotlib as plt
import numpy as np
from keras.saving.save import load_model
from tensorflow_core.python import confusion_matrix


class evaluate_mobilenetV2:
    def __init__(self, X_test, Y_test):
        self.x_test = X_test
        self.y_test = Y_test
        self.model = None
        self.BS = None

    def evaluate_model(self):
        self.model = load_model('saved_model/mobilenet_v1.h5')
        accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('\n', 'Test_Accuracy:-', accuracy[1])

    def plot_confusion_matrix(self):
        rounded_predictions = self.model.predict(self.x_test, batch_size=16, verbose=0)
        pred = np.argmax(rounded_predictions, axis=1)
        rounded_labels = np.argmax(self.y_test, axis=1)

        pred_Y = self.model.predict(self.x_test, batch_size=16, verbose=True)
        BS = 16

        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            target_names = ['BCC', 'Melanoma', 'Nevus']

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        # Predict the values from the validation dataset
        Y_pred = self.model.predict(self.x_test, batch_size=BS)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(pred_Y, axis=1)
        # Convert validation observations to one hot vectors
        # compute the confusion matrix
        rounded_labels = np.argmax(self.y_test, axis=1)
        confusion_mtx = confusion_matrix(rounded_labels, Y_pred_classes)

        # plot the confusion matrix
        plot_confusion_matrix(confusion_mtx, classes=range(3))

    def calculate_matrix(self):
        predIdxs = self.model.predict(self.x_test, batch_size=self.BS)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)
        rounded_labels = np.argmax(self.y_test, axis=1)

        # show a nicely formatted classification report
        print(classification_report(rounded_labels, predIdxs, target_names=['BCC', 'Melanoma', 'Nevus']))