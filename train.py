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
    epochs = 100


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


@ex.automain
def train(dataset, lr, epochs, _run, _log):
    train_file = os.path.join('data', dataset, 'data.pt')
    dataset = TensorDataset(torch.load(train_file))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VAE(im_size=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 1

    steps = 0
    log = '[{:d}/{:d}] MSE: {:.6f}  KL: {:.6f}  Total: {:.6f}'
    for epoch in range(1, epochs + 1):
        _log.info('Epoch {:d}'.format(epoch))

        train_loss = 0
        train_mse = 0
        train_kl = 0
        for i, d in enumerate(loader):
            steps += 1
            batch = d[0].to(device)
            optimizer.zero_grad()
            mse_loss, kl, out = model(batch)
            loss = mse_loss + kl
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse += mse_loss.item()
            train_kl += kl.item()
            if (i + 1) % log_every == 0:
                train_loss /= log_every
                train_mse /= log_every
                train_kl /= log_every
                _log.info(log.format(i + 1, len(loader), train_mse, train_kl,
                                     train_loss))
                _run.log_scalar('loss/total', train_loss, steps)
                _run.log_scalar('loss/mse', train_mse, steps)
                _run.log_scalar('loss/kl', train_kl, steps)
                train_loss = 0
                train_mse = 0
                train_kl = 0

        plot_examples(batch.cpu(), 'original')
        plot_examples(out.cpu().detach(), 'reconstruction')

    torch.save(model.state_dict(), f'vae.pt')
