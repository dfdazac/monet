import os
import sacred
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import VAE, MONet
from utils import (make_data_iterator, add_observers, train_ingredient,
                   plot_examples)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ex = sacred.Experiment(ingredients=[train_ingredient])
add_observers(ex)


# noinspection PyUnusedLocal
@ex.config
def config():
    model = 'monet'
    dataset = 'circles'
    decoder = 'deconv'
    beta = 2.0
    gamma = 0.25
    lr = 1e-4
    steps = 200000


@ex.automain
def train(model, dataset, decoder, beta, gamma, lr, steps, _run, _log):
    if len(_run.observers) == 0:
        _log.warning('Running without observers')

    train_file = os.path.join('data', dataset, 'data.pt')
    data = TensorDataset(torch.load(train_file))
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=1)
    iterator = make_data_iterator(loader)

    if model == 'vae':
        model = VAE(im_size=64, decoder=decoder, in_channels=3)
    elif model == 'monet':
        model = MONet(im_size=64, steps=5, beta=beta, gamma=gamma)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 1
    save_every = 20000
    train_loss = 0
    train_mse = 0
    train_kl = 0
    # log = '[{:d}/{:d}] MSE: {:.6f}  KL: {:.6f}  Loss: {:.6f}'
    log = '[{:d}/{:d}] Loss: {:.6f}'

    for step in range(1, steps + 1):
        # Train
        batch = next(iterator).to(device)
        # mse_loss, kl, out = model(batch)
        # loss = mse_loss + beta * kl
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        train_loss += loss.item()
        # train_mse += mse_loss.item()
        # train_kl += kl.item()
        if step % log_every == 0:
            train_loss /= log_every
            # train_mse /= log_every
            # train_kl /= log_every
            # _log.info(log.format(step, steps, train_mse, train_kl, train_loss))
            _log.info(log.format(step, steps, train_loss))
            # _run.log_scalar('mse', train_mse, step)
            # _run.log_scalar('kl', train_kl, step)
            # _run.log_scalar('loss', train_loss, step)
            train_loss = 0
            # train_mse = 0
            # train_kl = 0

        # Save
        # if step % save_every == 0:
        #     plot_examples(batch.cpu(), f'original_{step:d}')
        #     plot_examples(out.cpu().detach(), f'reconstruction_{step:d}')

    model_file = f'vae_{dataset}.pt'
    torch.save(model.state_dict(), model_file)
    _run.add_artifact(model_file)
    os.remove(model_file)
