import os
import matplotlib.pyplot as plt
import numpy as np
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
def plot_examples(examples, name=None, num_cols=8, _run=None):
    clipped = torch.clamp(examples.detach().cpu(), 0, 1)
    image = make_grid(clipped, nrow=num_cols, pad_value=1)
    c, h, w = image.shape
    fig = plt.gcf()
    fig.set_size_inches(0.01 * w, 0.01 * h)
    plt.cla()
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(name)

    if _run is not None:
        img_path = name + '.png'
        plt.savefig(img_path)
        _run.add_artifact(img_path, img_path)
        os.remove(img_path)


def generate_spaced_coordinates(low, high, num_points, levels):
    x = []
    y = []
    labels = []
    values = np.linspace(0, levels, num_points, endpoint=True)
    for i in values:
        for j in values:
            if i % 1 == 0 or j % 1 == 0:
                x.append(j)
                y.append(i)
                if j % 1 == 0:
                    labels.append(np.floor(j))
                elif i % 1 == 0:
                    labels.append(np.floor(i) + levels + 5)

    coords = np.stack((x, y), axis=0)
    coords = coords / levels * (high - low) + low
    x, y = coords

    return x, y, labels
