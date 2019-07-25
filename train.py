import os
from collections import defaultdict
import sacred
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import MONet
from utils import (make_data_iterator, add_observers, train_ingredient,
                   plot_examples)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ex = sacred.Experiment(name='MONet', ingredients=[train_ingredient])
add_observers(ex)

# TODO: Train VAE with monochromatic dataset!

# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'circles'
    num_slots = 5
    beta = 0.5
    gamma = 0.25
    lr = 1e-4
    steps = 200000


@ex.automain
def train(dataset, num_slots, beta, gamma, lr, steps, _run, _log):
    if len(_run.observers) == 0:
        _log.warning('Running without observers')

    train_file = os.path.join('data', dataset, 'data.pt')
    data = TensorDataset(torch.load(train_file))
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=1)
    iterator = make_data_iterator(loader)
    # FIXME
    im_channels = 1  # next(iter(loader))[0].shape[1]

    model = MONet(im_size=64, im_channels=im_channels, num_slots=num_slots)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 200
    save_every = 1000
    metrics = defaultdict(float)

    for step in range(1, steps + 1):
        # Train
        batch = next(iterator).to(device)[:, 0:1]
        mse, kl, mask_kl, recs, log_masks = model(batch)
        loss = mse + beta * kl + gamma * mask_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        metrics['mse'] += mse.item()
        metrics['kl'] += kl.item()
        metrics['mask_kl'] += mask_kl.item()
        metrics['loss'] += loss.item()
        if step % log_every == 0:
            log = f'[{step:d}/{steps:d}] '
            for m in metrics:
                metrics[m] /= log_every
                log += f'{m}: {metrics[m]:.6f} '
                _run.log_scalar(m, metrics[m], step)
                metrics[m] = 0.0

            _log.info(log)

        # Save
        if step % save_every == 0:
            recs = recs.reshape(-1, im_channels, 64, 64)
            log_masks = torch.exp(log_masks).reshape(-1, 1, 64, 64)
            num_cols = batch.shape[0]

            plot_examples(batch, f'original_{step:d}')
            plot_examples(recs, f'reconstruction_{step:d}', num_cols)
            plot_examples(log_masks, f'mask_{step:d}', num_cols)

    model_file = f'monet_{dataset}.pt'
    torch.save(model.state_dict(), model_file)
    _run.add_artifact(model_file)
    os.remove(model_file)

