import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sacred

from models import VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ex = sacred.Experiment()


# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'circles'
    lr = 1e-4
    steps = 100


@ex.capture
def plot_examples(examples, name, _run):
    clipped = torch.clamp(examples.detach(), 0, 1)
    image = make_grid(clipped)
    plt.cla()
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')

    img_path = name + '.png'
    plt.savefig(img_path)
    _run.add_artifact(img_path, img_path)
    os.remove(img_path)


def make_data_iterator(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)[0].to(device)
        except StopIteration:
            iterator = iter(loader)
            continue


@ex.automain
def train(dataset, lr, steps, _run, _log):
    train_file = os.path.join('data', dataset, 'data.pt')
    dataset = TensorDataset(torch.load(train_file))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    iterator = make_data_iterator(loader)

    model = VAE(im_size=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 1
    save_every = 100
    train_loss = 0
    train_mse = 0
    train_kl = 0
    log = '[{:d}/{:d}] MSE: {:.6f}  KL: {:.6f}  Loss: {:.6f}'

    for step in range(1, steps + 1):
        # Train
        batch = next(iterator)
        mse_loss, kl, out = model(batch)
        loss = mse_loss + kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        train_loss += loss.item()
        train_mse += mse_loss.item()
        train_kl += kl.item()
        if step % log_every == 0:
            train_loss /= log_every
            train_mse /= log_every
            train_kl /= log_every
            _log.info(log.format(step, steps, train_mse, train_kl, train_loss))
            _run.log_scalar('mse', train_mse, step)
            _run.log_scalar('kl', train_kl, step)
            _run.log_scalar('loss', train_loss, step)
            train_loss = 0
            train_mse = 0
            train_kl = 0

        # Save
        if step % save_every == 0:
            plot_examples(batch.cpu(), f'original_{step:d}')
            plot_examples(out.cpu().detach(), f'reconstruction_{step:d}')
            torch.save(model.state_dict(), f'vae.pt')
