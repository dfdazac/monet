import os
from collections import defaultdict
import sacred
import torch
from torch.utils.data import TensorDataset, DataLoader
from sacred.utils import SacredInterrupt

from models import MONet
import utils


class TrainingFailed(SacredInterrupt):
    STATUS = 'TRAIN_FAILED'


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
    gamma = 0.25
    lr = 1e-4
    steps = 200000


@ex.automain
def train(dataset, num_slots, z_dim, beta, gamma, lr, steps, _run, _log):
    if len(_run.observers) == 0:
        _log.warning('Running without observers')

    train_file = os.path.join('data', dataset, 'data.pt')
    data = TensorDataset(torch.load(train_file))
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=1)
    iterator = utils.make_data_iterator(loader)
    _, im_channels, im_size, _ = next(iter(loader))[0].shape

    model = MONet(im_size, im_channels, num_slots, z_dim).to(device)

    model.load_state_dict(torch.load('monet_sprites_multi.pt', map_location='cpu'))
    batch = torch.load('bad_batch.pt')

    optimizer = torch.optim.Adam(model.parameters(), lr)

    log_every = 1
    save_every = 10000
    metrics = defaultdict(float)
    successful = True

    try:
        for step in range(1, steps + 1):
            # Train
            # batch = next(iterator).to(device)

            with torch.autograd.detect_anomaly():
                mse, kl, mask_kl, recs, log_masks = model(batch)
                loss = mse + beta * kl + gamma * mask_kl
                optimizer.zero_grad()
                loss.backward()

            max_grad = torch.tensor(0.0).to(device)
            for param in model.parameters():
                grad = torch.max(param.grad)
                max_grad = torch.max(max_grad, grad)
                assert not torch.isnan(max_grad), 'nan in grad'

            optimizer.step()

            # Log
            metrics['mse'] += mse.item()
            metrics['kl'] += kl.item()
            metrics['mask_kl'] += mask_kl.item()
            metrics['loss'] += loss.item()
            metrics['max_grad'] += max_grad.item()
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

                utils.plot_examples(batch, f'original_{step:d}', num_cols)
                utils.plot_examples(recs, f'reconstruction_{step:d}', num_cols)
                utils.plot_examples(log_masks, f'mask_{step:d}', num_cols)

    except AssertionError as error:
        _log.error(error)
        successful = False
        batch_file = 'bad_batch.pt'
        torch.save(batch.cpu(), batch_file)
        _run.add_artifact(batch_file)
        # os.remove(batch_file)

    model_file = f'monet_{dataset}.pt'
    torch.save(model.state_dict(), model_file)
    _run.add_artifact(model_file)
    os.remove(model_file)

    if not successful:
        raise TrainingFailed()
