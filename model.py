import os

import attr
import click


def _search_driving_logs(self, attribute, value):
    found = False
    for root, dirs, files in os.walk(value, followlinks=False):
        for name in files:
            if name == "driving_log.csv":
                found = True
    if not found:
        raise ValueError("No driving_log found")


@attr.s
class DrivingLogs:
    base_path = attr.ib(default=".",
                        validator=_search_driving_logs)

    @property
    def driving_logs(self):
        return self._iterate_driving_logs()

    def _iterate_driving_logs(self):
        for root, dirs, files in os.walk(self.base_path, followlinks=False):
            for name in files:
                if name == "driving_log.csv":
                    my_path = os.path.abspath(os.path.join(root, name))
                    yield my_path

@click.command()
@click.option('--driving_logs', default=".",
              help='The roo path containing the driving logs and the corresponding images',
              type=click.Path(exists=True))
def cli(driving_logs):
    click.echo(DrivingLogs(driving_logs).driving_logs)


if __name__ == '__main__':
    cli()
