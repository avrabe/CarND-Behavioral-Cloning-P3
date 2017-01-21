import click


@click.command()
@click.option('--driving_logs', default=".",
              help='The roo path containing the driving logs and the corresponding images',
              type=click.Path(exists=True))
def cli(driving_logs):
    click.echo(driving_logs)


if __name__ == '__main__':
    cli()
