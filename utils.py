import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from sacred.observers import MongoObserver
from sacred import Ingredient


def make_data_iterator(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            continue


def add_observers(experiment):
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        experiment.observers.append(MongoObserver.create(uri, database))


train_ingredient = Ingredient('utils')


@train_ingredient.capture
def plot_examples(examples, name, _run=None):
    clipped = torch.clamp(examples.detach(), 0, 1)
    image = make_grid(clipped, pad_value=1)
    plt.cla()
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(name)

    if _run is not None:
        img_path = name + '.png'
        plt.savefig(img_path)
        _run.add_artifact(img_path, img_path)
        os.remove(img_path)
