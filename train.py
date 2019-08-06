import os
from collections import defaultdict
import sacred
import torch
from torch.utils.data import TensorDataset, DataLoader
from sacred.utils import SacredInterrupt

from models import MONet
import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ex = sacred.Experiment(name='MONet', ingredients=[utils.train_ingredient])
utils.add_observers(ex)

# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'circles'
    num_slots = 5
    z_dim = 16
    beta = 0.5
    gamma = 0.5
    lr = 1e-4
    batch_size = 16
    steps = 200000


@ex.automain
def train(dataset, num_slots, z_dim, beta, gamma, lr, batch_size,
          steps, _run, _log):
    if len(_run.observers) == 0:
        _log.warning('Running without observers')

    train_file = os.path.join('data', dataset, 'data.pt')
    data = TensorDataset(torch.load(train_file))
    loader = DataLoader(data, batch_size, shuffle=True, num_workers=1,
                        drop_last=True)
    iterator = utils.make_data_iterator(loader)
    _, im_channels, im_size, _ = next(iter(loader))[0].shape

    model = MONet(im_size, im_channels, num_slots, z_dim).to(device)

    # model.load_state_dict(torch.load('monet_sprites_multi.pt',
    #                                  map_location='cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 500
    save_every = 10000
    max_samples = 16
    metrics = defaultdict(float)

    for step in range(1, steps + 1):
        # Train
        batch = next(iterator).to(device)
        nll, kl, mask_kl, recs, masks = model(batch)
        loss = nll + beta * kl + gamma * mask_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        metrics['nll'] += nll.item()
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
            x = batch[:max_samples]
            recs = recs[:max_samples]
            masks = masks[:max_samples]

            final = torch.sum(recs * masks, dim=1)
            recs = recs.reshape(-1, im_channels, im_size, im_size)
            masks = masks.reshape(-1, 1, im_size, im_size)

            utils.plot_examples(x, f'original_{step:d}', num_cols=1)
            utils.plot_examples(recs, f'reconstruction_{step:d}', num_slots)
            utils.plot_examples(masks, f'mask_{step:d}', num_slots)
            utils.plot_examples(final, 'final', num_cols=1)

    model_file = f'monet_{dataset}.pt'
    torch.save(model.state_dict(), model_file)
    _run.add_artifact(model_file)
    os.remove(model_file)
