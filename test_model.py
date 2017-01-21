import os.path

import pytest

import model


def test_all_driving_logs_csv():
    dl = model.DrivingLogs()
    assert len(list(dl.driving_logs)) == 2


def test_one_driving_logs_csv():
    dl = model.DrivingLogs(os.path.join("test_data", "one"))
    assert len(list(dl.driving_logs)) == 1


def test_no_driving_logs_csv():
    with pytest.raises(ValueError):
        model.DrivingLogs(os.path.join("test_data", "empty"))
