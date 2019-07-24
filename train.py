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


# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'circles'
    num_slots = 5
    beta = 2.0
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

    model = MONet(im_size=64, im_channels=3, num_slots=num_slots)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 1
    save_every = 20
    metrics = defaultdict(float)

    for step in range(1, steps + 1):
        # Train
        batch = next(iterator).to(device)
        mse, kl, mask_kl, out = model(batch)
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
                _run.log_scalar(m, metrics[m])
                metrics[m] = 0.0

            _log.info(log)

        # Save
        if step % save_every == 0:
            plot_examples(batch.cpu(), f'original_{step:d}')
            out = out.reshape(-1, 3, 64, 64)
            plot_examples(out.cpu(), f'reconstruction_{step:d}',
                          num_cols=num_slots)

    model_file = f'{model}_{dataset}.pt'
    torch.save(model.state_dict(), model_file)
    _run.add_artifact(model_file)
    os.remove(model_file)

