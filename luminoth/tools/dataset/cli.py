import click

from .voc import voc
from .coco import coco


@click.group(help='Groups of commands to manage datasets')
def dataset():
    pass


dataset.add_command(voc)
dataset.add_command(coco)
