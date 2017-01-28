import json
import os
import sys
from itertools import product

import attr
import click
import cv2
import numpy as np
import pandas as pd
from keras.layers import Convolution2D, Dropout
from keras.layers import Dense, Flatten, Lambda, ELU
from keras.models import Sequential
from keras.utils.visualize_util import model_to_dot
from sklearn.model_selection import train_test_split as tts
from sklearn.utils import shuffle


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

    @property
    def train_test_split(self):
        images = self.data_frame['center'].append(self.data_frame['left']).append(self.data_frame['right'])
        steering = self.data_frame['steering'].append(self.data_frame['steering']).append(self.data_frame['steering'])
        return tts(images, steering)


def generate_arrays_from_file(x, y, batch_size, do_shuffle=False):
    batch_count = 0
    # batch_size /= 10
    while 1:
        batch_index = 0
        if batch_index == 0:
            X = []
            Y = []
        if do_shuffle:
            x, y = shuffle(x, y)
        for filename, steering in zip(x, y):
            # create numpy arrays of input data
            # and labels, from each line in the file
            img = cv2.imread(filename)
            if img is None:
                print("Skip unreadable %s" % filename)
                break
            img = cv2.resize(img, (200, 66))
            X.append(np.copy(img))
            Y.append(steering)
            # print(steering, filename, img.shape)
            batch_index += 1
            if batch_index == batch_size:
                # print("BatchCount %d" % (batch_count))
                batch_count += 1
                # foo = X, Y
                """
                Below line I got from
                https://groups.google.com/forum/#!topic/keras-users/of7puzB2H0g
                """
                foo = np.asarray(X), np.asarray(Y)
                yield foo
                batch_index = 0
                X = []
                Y = []


@attr.s
class Filename:
    model = attr.ib(default="unknown")
    optimizer = attr.ib(default="unknown")
    objective = attr.ib(default="unknown")
    epoch = attr.ib(default=0)
    samples_per_epoch = attr.ib(default=0)
    batch_size = attr.ib(default=0)
    validate = attr.ib(default=False)

    def get_filename(self, suffix=".json"):
        return "%s_%s_%s_e%02d_s%06d_b%04d_v%s%s" % (self.model, self.optimizer,
                                                     self.objective, self.epoch, self.samples_per_epoch,
                                                     self.batch_size, self.validate, suffix)


def commaai():
    """
    Provide the commaai model described in
    https://github.com/commaai/research/blob/master/train_steering_model.py
    The input sizes have been adjusted.
    :return A keras model:
    """
    ch, row, col = 3, 66, 200  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model


def nvidia():
    model = Sequential()
    model.add(
        Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), input_shape=(66, 200, 3), activation="relu"))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1164, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    return model


@click.command()
@click.option('--models', default=["nvidia"], multiple=True,
              type=click.Choice(["nvidia", "commaai"]),
              help='Select the model.')
@click.option('--optimizers', default=["adam"], multiple=True,
              type=click.Choice(["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]),
              help='Select the optimizer.')
@click.option('--objectives', default=["mse"], multiple=True,
              type=click.Choice(["mse", "mae", "mape", "msle", "squared_hinge", "hinge"]),
              help='Select the objective')
@click.option('-e', '--epochs', default=[1], multiple=True, type=int,
              help='How many epochs should be trained.')
@click.option('--samples_per_epoch', default=15000, type=int,
              help='How many samples per epoch.')
@click.option('--batch_size', default=64, type=int,
              help='Whats the batch size.')
@click.option('--validate/--no-validate', default=True)
@click.option('-l', '--driving_logs', default=".",
              help='The root path containing the driving logs and the corresponding images',
              type=click.Path(exists=True))
def cli(models, optimizers, objectives, epochs, samples_per_epoch, batch_size, validate, driving_logs):
    X_train, X_val, y_train, y_val = DrivingLogs(driving_logs).train_test_split
    print(len(X_train), len(X_val))

    for m, optimizer, objective, nb_epoch in product(models, optimizers, objectives, epochs):
        print("=== %s ===== %s ===== %s ===== %02d =====" % (m, optimizer, objective, nb_epoch))
        my_filename = Filename(model=m, optimizer=optimizer, objective=objective, epoch=nb_epoch,
                               samples_per_epoch=samples_per_epoch, batch_size=batch_size,
                               validate=validate)
        this_module = sys.modules[__name__]
        getattr(this_module, m)
        model = getattr(this_module, m)()

        model.compile(loss=objective,
                      optimizer=optimizer,
                      metrics=["mean_squared_error"])

        nb_val_samples = 2000

        with open(my_filename.get_filename(suffix=".dot"), 'w') as jfile:
            jfile.write(model_to_dot(model, show_shapes=True, show_layer_names=True).to_string())
        if validate:
            history = model.fit_generator(generate_arrays_from_file(X_train, y_train, batch_size, do_shuffle=True),
                                          samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                                          validation_data=generate_arrays_from_file(X_val, y_val, batch_size),
                                          nb_val_samples=nb_val_samples)
            print("The val_mean_squared_error is: %.3f" % history.history['val_mean_squared_error'][-1])
            print("The loss is: %.3f" % history.history['val_loss'][-1])
        else:
            model.fit_generator(generate_arrays_from_file(X_train, y_train, batch_size, do_shuffle=True),
                                samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch)

        with open(my_filename.get_filename(), 'w') as jfile:
            json.dump(model.to_json(), jfile)
        model.save_weights(my_filename.get_filename(suffix=".h5"), overwrite=True)

        myfile = "C:\\Users\\Ralf\\Downloads\\simulator-windows-64\\left_route\\IMG\\center_2017_01_19_19_25_49_380.jpg"
        img = cv2.imread(myfile)
        img = cv2.resize(img, (200, 66))

        X = np.asarray([np.copy(img)])
        steering_angle = float(model.predict(X, batch_size=1, verbose=1))
        print(optimizer, objective, nb_epoch, Filename, "-0.307052", steering_angle)


if __name__ == '__main__':
    cli()
