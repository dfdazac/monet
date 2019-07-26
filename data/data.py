from collections import namedtuple
from PIL import Image, ImageDraw
import os.path as osp
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sacred

ex = sacred.Experiment()


# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'circles'
    num_samples = 50000


IMG_SIZE = 64
Sprite = namedtuple('Sprite', ['shape', 'x', 'y', 'size', 'color'])


def draw_sprites(sprites, img_size, bg_color):
    """Draw an image given a list of sprites.

    Args:
        sprites (list): contains named tuples of type Shape
        img_size (int): side length of the image
        bg_color (tuple): rgb color for the background

    Returns:
        torch.tensor of shape (3, img_size, img_size)
    """
    img = Image.new('RGB', (img_size, img_size), color=bg_color)
    drawer = ImageDraw.Draw(img)

    for sprite in sprites:
        shape, x0, y0, size, color = sprite
        x1 = x0 + size
        y1 = y0 + size
        box = [x0, y0, x1, y1]

        if shape == 'polygon':
            box[0] = (x1 + x0) / 2
            box += [x0, y1]

        getattr(drawer, shape)(box, fill=color)

    img_tensor = torch.tensor(np.array(img), dtype=torch.float)
    # Move channels first and normalize to values in [0, 1]
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0

    return img_tensor


def generate_random_img(img_size, num_sprites, shapes, colors, bg_color,
                        size=None):
    """Generate an image with random sprites in it.

    Args:
        img_size (int): side length of the image
        num_sprites (int): number of sprites to draw
        shapes (list): shapes (str) to choose from
        colors (list): colors (rgb tuple) to choose from
        bg_color (tuple): rgb color for the background
        size (int): sprite size. If None, each sprite has a random size.
            Default: None
    Returns:
        PIL.Image
    """
    idx_color_sprite = np.arange(len(colors))
    np.random.shuffle(idx_color_sprite)

    if size is None:
        sizes = np.random.randint(low=10, high=25, size=num_sprites)
    else:
        sizes = np.ones(num_sprites) * size

    x_all = np.random.uniform(0, img_size - sizes, size=num_sprites)
    y_all = np.random.uniform(0, img_size - sizes, size=num_sprites)
    shape_idx = np.random.choice(np.arange(len(shapes)),
                                 size=num_sprites, replace=False)

    sprites = []

    for i in range(num_sprites):
        x, y = x_all[i], y_all[i]
        shape = shapes[shape_idx[i]]
        sprites.append(Sprite(shape, x, y, sizes[i],
                              colors[idx_color_sprite[i]]))

    img_tensor = draw_sprites(sprites, img_size, bg_color)

    return img_tensor


def generate_dataset(name, num_samples, max_sprites, shapes, colors, bg_color,
                     num_channels=3, size=None):
    """Generate a dataset of images and save it to disk.
        The saved tensor has shape (num_samples, 3, img_size, img_size).

        Args:
            name (str): name of the folder where tensor will be stored
            num_samples (int): number of samples
            max_sprites (int): Maximum number of sprites to draw. For each
                image, the actual number is randomly sampled in
                [1, max_sprites]
            shapes (list): shapes (str) to choose from
            colors (list): colors (rgb tuple) to choose from
            bg_color (tuple): rgb color for the background
            num_channels (int): number of color channels in the dataset
            size (int): sprite size. If None, each sprite has a random size.
                Default: None
    """
    print('Generating dataset...')
    np.random.seed(0)
    data = torch.empty([num_samples, num_channels, IMG_SIZE, IMG_SIZE],
                       dtype=torch.float32)

    for i in range(num_samples):
        num_sprites = np.random.randint(low=1, high=max_sprites + 1)
        data[i] = generate_random_img(IMG_SIZE, num_sprites,
                                      shapes,
                                      colors,
                                      bg_color,
                                      size)[0:num_channels]

    path = osp.join(name, 'data.pt')
    torch.save(data, path)
    print(f'Saved dataset to {path}')


@ex.command(unobserved=True)
def generate_circles(num_samples):
    """Generate samples of images with circles and save tensor to disk.
    The saved tensor has shape (num_samples, 3, img_size, img_size).

    Args:
        num_samples (int): number of samples
    """
    generate_dataset('circles', num_samples, max_sprites=1, shapes=['ellipse'],
                     colors=[(255, 255, 255)], bg_color=(0, 0, 0),
                     num_channels=1, size=10)


@ex.command(unobserved=True)
def generate_sprites_single(num_samples):
    """Generate samples of images with circles and save tensor to disk.
    The saved tensor has shape (num_samples, 3, img_size, img_size).

    Args:
        num_samples (int): number of samples
    """
    colors = [(0, 0, 210),
              (0, 210, 0),
              (210, 0, 0),
              (150, 150, 0),
              (150, 0, 150),
              (0, 150, 150)]

    shapes = ['ellipse', 'rectangle', 'polygon']
    generate_dataset('sprites', num_samples, max_sprites=1, shapes=shapes,
                     colors=colors, bg_color=(255, 255, 255))


@ex.command(unobserved=True)
def generate_sprites_multi(num_samples):
    """Generate samples of images with circles and save tensor to disk.
    The saved tensor has shape (num_samples, 3, img_size, img_size).

    Args:
        num_samples (int): number of samples
    """
    colors = [(0, 0, 210),
              (0, 210, 0),
              (210, 0, 0),
              (150, 150, 0),
              (150, 0, 150),
              (0, 150, 150)]

    shapes = ['ellipse', 'rectangle', 'polygon']
    generate_dataset('sprites_multi', num_samples, max_sprites=3,
                     shapes=shapes, colors=colors, bg_color=(255, 255, 255))


@ex.command(unobserved=True)
def show_samples(dataset, num_samples):
    dataset = torch.load(osp.join(dataset, 'data.pt'))
    grid = make_grid(dataset[:num_samples]).permute(1, 2, 0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    ex.run_commandline()
