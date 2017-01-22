import os

import attr
import click
import cv2
import numpy as np
import pandas as pd
from keras.layers import Convolution2D
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split as tts


def _convert_image_filename(filename, path):
    """
    Could be absolute or relative with
    separators from different operating systems. In general an image will
    look like .*IMG<os.seperator>.*. As of this everything before the IMG
    will be removed and the original os.seperator estimated from the first
    character after IMG.
    :param filename: The original image path
    :param path: The new path the pictures are located
    :return: The new absolute filename
    """
    result = filename[filename.find("IMG"):]
    orig_os_path_set = result[3]
    result = result.replace(orig_os_path_set, os.path.sep)
    result = os.path.join(path, result)
    result = os.path.abspath(result)
    return result


def _read_driving_log(filename):
    """
    Read a driving log file. It needs to contain following format:
    - comma separated, with or without header, decimal is .
    The content is
    - center image: string representing a file.
    - left image: string representing a file.
    - right image: string representing a file.
    - steering angle: float between -1 and +1.
    - throttle: float
    - break: float
    - speed: float
    The first row will be skipped to avoid data which contains headers.
    :param filename: abso
    :return:
    """
    csv = pd.read_csv(filename, sep=",", header=None,
                      names=["center", "left", "right",
                             "steering", "throttle",
                             "break", "speed"],
                      dtype={"center": object, "left": object, "right": object,
                             "steering": np.float64, "throttle": np.float64,
                             "break": np.float64, "speed": np.float64},
                      decimal=".", skiprows=1)
    for index in ["center", "left", "right"]:
        for i in csv.index:
            name = csv.ix[i, index]
            csv.set_value(i, index, _convert_image_filename(name, os.path.dirname(filename)))

    return csv


def _iterate_through_path_to_search_for(path, filename):
    """
    Scan through a directories starting with path and return all found files
    with name filename
    :param path: The root directory to start the scan.
    :param filename: The filename to search for.
    :return: An iterator containing the absolute path of the found filename
    """
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            if name == filename:
                my_path = os.path.abspath(os.path.join(root, name))
                yield my_path


def _search_driving_logs(self, attribute, value):
    """
    Check if the provided path contains driving log csv files
    :param self: Ignored.
    :param attribute: Ignored
    :param value: The root path to search for the driving logs
    :return: A ValueError if no logs can be found otherwise nothing
    """
    files = list(_iterate_through_path_to_search_for(value, "driving_log.csv"))
    if len(files) == 0:
        raise ValueError("No driving_log found")


@attr.s
class DrivingLogs:
    """
    """
    base_path = attr.ib(default=".",
                        validator=_search_driving_logs)

    @property
    def driving_logs(self):
        """
        :return: An iterator with all the driving_logs files found.
        """
        return _iterate_through_path_to_search_for(self.base_path, "driving_log.csv")

    @property
    def data_frame(self):
        """

        :return: Return a panda Data Frame with all the
        """
        return pd.concat([_read_driving_log(x) for x in self.driving_logs])


def generate_arrays_from_file(x, y):
    batchSize = 1
    while 1:
        for filename, steering in zip(x, y):
            # create numpy arrays of input data
            # and labels, from each line in the file
            X = np.zeros((batchSize, 160, 320, 3))
            Y = np.zeros((batchSize, 1))

            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # print(steering, filename, img.shape)
            X[0] = img
            Y[0] = steering

            foo = X, Y
            yield foo


@click.command()
@click.option('--driving_logs', default=".",
              help='The roo path containing the driving logs and the corresponding images',
              type=click.Path(exists=True))
def cli(driving_logs):
    data = DrivingLogs(driving_logs).data_frame
    print(len(data))
    X_train, X_val, y_train, y_val = tts(data['center'], data['steering'])

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(160, 320, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 2, 2))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 2, 2))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 2, 2))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 2, 2))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 2, 2))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(400))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='mse',
                  optimizer='adam')

    history = model.fit_generator(generate_arrays_from_file(X_train, y_train),
                                  samples_per_epoch=300, nb_epoch=40,
                                  validation_data=generate_arrays_from_file(X_val, y_val),
                                  nb_val_samples=100)

    print("The validation accuracy is: %.3f" % history.history['val_acc'][-1])


if __name__ == '__main__':
    cli()
