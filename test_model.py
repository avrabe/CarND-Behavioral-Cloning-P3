import os
import os.path

import pytest

import model
from model import DrivingLogs, ModelOptions, train_model


def test_all_driving_logs_csv():
    dl = model.DrivingLogs()
    assert len(list(dl.driving_logs)) == 3


def test_one_driving_logs_csv():
    dl = model.DrivingLogs(os.path.join("test_data", "one"))
    assert len(list(dl.driving_logs)) == 1


def test_no_driving_logs_csv():
    with pytest.raises(ValueError):
        model.DrivingLogs(os.path.join("test_data", "empty"))


def test_empty_driving_logs_csv():
    with pytest.raises(Exception):
        model.DrivingLogs(os.path.join("test_data", "two")).data_frame()


def test_incorrect_driving_logs_csv():
    with pytest.raises(Exception):
        model.DrivingLogs(os.path.join("test_data", "three")).data_frame()


def test_incorrect_driving_logs_csv():
    with pytest.raises(Exception):
        model.DrivingLogs(os.path.join("test_data", "one")).data_frame()


def test_linux_image_filename_conversion():
    if os.name == "nt":
        result = "C:\\a\\b\\IMG\\foo.jpg"
        path = "C:\\a\\b\\"
    elif os.name == "posix":
        result = "/a/b/IMG/foo.jpg"
        path = "/a/b/"
    else:
        assert "Unsupported system" == 1
    assert model._convert_image_filename("/x/y/IMG/foo.jpg", path) == result
    assert model._convert_image_filename("IMG/foo.jpg", path) == result

        
def test_train_model():
    X_train, X_val, y_train, y_val = DrivingLogs("/tmp/data").train_validation_split
    model_options = ModelOptions(model="nvidia", optimizer="adam", objective="mse", epoch=1,
                                 samples_per_epoch=10, batch_size=1,
                                 validate=True, validation_samples_per_epoch=10)
    m = train_model(X_train, X_val, y_train, y_val, model_options)
    assert not m == None
