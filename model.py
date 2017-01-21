import os

import attr
import click
import pandas as pd


def _read_driving_log(filename, validate=False):
    """
    Read a driving log file. It needs to contain following format:
    - comma separated
    The content is
    - center image (string representing a file)
    - left image (string representing a file)
    - right image (string representing a file)
    - steering angle (float between -1 and +1)
    - throttle (TBD)
    - break (TBD)
    - speed (TBD)
    :param filename: The dr
    :return:
    """
    csv = pd.read_csv(filename, sep=",")
    if validate:
        pass
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


@click.command()
@click.option('--driving_logs', default=".",
              help='The roo path containing the driving logs and the corresponding images',
              type=click.Path(exists=True))
def cli(driving_logs):
    for i in DrivingLogs(driving_logs).driving_logs:
        click.echo(_read_driving_log(i).to_string())


if __name__ == '__main__':
    cli()
