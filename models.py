import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence


class VAE(nn.Module):
    """Variational Autoencoder with spatial broadcast decoder, or
    deconvolutional decoder.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, H_{in}, W_{in})`
    """
    def __init__(self, im_size, in_channels, z_dim, decoder='sbd'):
        super(VAE, self).__init__()

        enc_convs = [nn.Conv2d(in_channels, out_channels=64,
                               kernel_size=4, stride=2, padding=1)]
        enc_convs.extend([nn.Conv2d(in_channels=64, out_channels=64,
                                    kernel_size=4, stride=2, padding=1)
                          for i in range(3)])
        self.enc_convs = nn.ModuleList(enc_convs)

        self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                nn.ReLU(),
                                nn.Linear(in_features=256,
                                          out_features=2 * z_dim))

        if decoder == 'deconv':
            self.dec_linear = nn.Linear(in_features=z_dim, out_features=256)
            dec_convs = [nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                            kernel_size=4, stride=2, padding=1)
                         for i in range(4)]
            self.dec_convs = nn.ModuleList(dec_convs)
            self.decoder = self.deconv_decoder
            self.last_conv = nn.ConvTranspose2d(in_channels=64,
                                                out_channels=in_channels,
                                                kernel_size=4, stride=2,
                                                padding=1)

        elif decoder == 'sbd':
            # Coordinates for the broadcast decoder
            self.im_size = im_size
            x = torch.linspace(-1, 1, im_size)
            y = torch.linspace(-1, 1, im_size)
            x_grid, y_grid = torch.meshgrid(x, y)
            # Add as constant, with extra dims for N and C
            self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
            self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

            dec_convs = [nn.Conv2d(in_channels=z_dim + 2, out_channels=64,
                                   kernel_size=3, padding=1),
                         nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=3, padding=1)]
            self.dec_convs = nn.ModuleList(dec_convs)
            self.decoder = self.sb_decoder
            self.last_conv = nn.Conv2d(in_channels=64,
                                       out_channels=in_channels,
                                       kernel_size=3, padding=1)

    def encoder(self, x):
        batch_size = x.size(0)
        for module in self.enc_convs:
            x = F.relu(module(x))

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return torch.chunk(x, 2, dim=1)

    def deconv_decoder(self, z):
        x = F.relu(self.dec_linear(z)).view(-1, 64, 2, 2)
        for module in self.dec_convs:
            x = F.relu(module(x))
        x = self.last_conv(x)

        return x

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sb_decoder(self, z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size, self.im_size)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        for module in self.dec_convs:
            x = F.relu(module(x))
        x = self.last_conv(x)

        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar)
        x_rec = self.decoder(z)

        return mu, logvar, x_rec

    def loss(self, x):
        batch_size = x.shape[0]
        mu, logvar, x_rec = self.forward(x)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        mse_loss = F.mse_loss(x_rec, x, reduction='none').view(batch_size, -1)
        mse_loss = 10 * mse_loss.sum(dim=-1).mean()

        return mse_loss, kl, x_rec


class UNetBlock(nn.Module):
    """Convolutional block for UNet, containing: conv -> instance-norm -> relu

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the block

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class UNet(nn.Module):
    """UNet, based on 'U-Net: Convolutional Networks for Biomedical
    Image Segmentation' by O. Ronneberger et al. It consists of contracting and
    expanding paths that at each block double and expand the size,
    respectively. Skip tensors are concatenated to the expanding path.
    A last 1x1 convolution reduces the number of channels to 1.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the network
        init_channels (int): Number of channels produced by the first block.
            This is doubled in subsequent blocks in the path. Default: 32
        depth (int): number of blocks in each path. Default: 3

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """
    def __init__(self, in_channels, out_channels, init_channels=32, depth=3):
        super(UNet, self).__init__()
        self.depth = depth

        self.down_blocks = nn.ModuleList()
        n_channels = init_channels
        for i in range(depth):
            self.down_blocks.append(UNetBlock(in_channels, n_channels))
            in_channels = n_channels
            n_channels *= 2
        n_channels //= 2

        mid_block = [UNetBlock(n_channels, n_channels * 2),
                     UNetBlock(n_channels * 2, n_channels)]
        self.mid_block = nn.Sequential(*mid_block)
        in_channels = 2 * n_channels
        n_channels //= 2

        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(UNetBlock(in_channels, n_channels))
            in_channels = 2 * n_channels
            n_channels //= 2
        n_channels *= 2

        self.last_conv = nn.Conv2d(n_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_tensors = []
        for i, module in enumerate(self.down_blocks):
            x = module(x)
            skip_tensors.append(x)
            x = F.interpolate(x, scale_factor=0.5)

        x = self.mid_block(x)

        for block, skip in zip(self.up_blocks, reversed(skip_tensors)):
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat((skip, x), dim=1)
            x = block(x)

        x = self.last_conv(x)

        return x


class AttentionNetwork(nn.Module):
    """A network that takes an image and a scope, to generate a mask for the
    part of the image that needs to be explained, and a scope for the next
    step.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` (image),
                 :math:`(N, 1, H_{in}, W_{in})` (scope)
        - Output: :math:`(N, 1, H_{out}, W_{out})` (mask),
                  :math:`(N, 1, H_{out}, W_{out})` (next scope)
    """
    def __init__(self, in_channels):
        super(AttentionNetwork, self).__init__()
        self.unet = UNet(in_channels, out_channels=1)

    def forward(self, x, log_scope):
        x = torch.cat((log_scope, x), dim=1)
        x = self.unet(x)
        log_mask = log_scope + F.logsigmoid(x)
        log_scope = log_scope + F.logsigmoid(-x)
        return log_mask, log_scope


def clamp_probs(logprobs):
    return torch.clamp(logprobs.exp(), min=1e-6, max=1 - 1e-6)


class MONet(nn.Module):
    def __init__(self, im_size, im_channels, num_slots, z_dim):
        super(MONet, self).__init__()

        self.component_vae = VAE(im_size, im_channels + 1, z_dim)
        self.attention = AttentionNetwork(in_channels=im_channels + 1)

        self.im_size = im_size
        self.im_channels = im_channels
        self.num_slots = num_slots

        init_scope = torch.zeros((1, 1, im_size, im_size))
        self.register_buffer('init_scope', init_scope)
        scale = torch.empty((1, 1, 1, 1)).fill_(0.1)
        self.register_buffer('scale', scale)

    def forward(self, x):
        batch_size = x.shape[0]
        log_scope = self.init_scope.expand(batch_size, -1, -1, -1)
        scale = self.scale.expand_as(x)

        recs = torch.empty(self.num_slots, batch_size, self.im_channels,
                           self.im_size, self.im_size).to(x.device)
        masks = torch.empty(self.num_slots, batch_size, 1,
                            self.im_size, self.im_size).to(x.device)
        logprobs = torch.empty(self.num_slots, batch_size).to(x.device)

        kl_sum = mask_kl_sum = 0.0

        for slot in range(self.num_slots):
            if slot < self.num_slots - 1:
                log_mask, log_scope = self.attention(x, log_scope)
            else:
                log_mask = log_scope

            vae_in = torch.cat((x, log_mask), dim=1)
            mu, logvar, vae_out = self.component_vae(vae_in)

            # Reconstructions
            x_rec, log_mask_rec = torch.split(vae_out, self.im_channels, dim=1)
            log_mask_rec = F.logsigmoid(log_mask_rec)

            # Masked component reconstruction loss
            rec_dist = Normal(x_rec, scale)
            rec_loss = log_mask + rec_dist.log_prob(x)
            rec_loss = rec_loss.view(batch_size, -1)
            rec_loss = rec_loss.sum(dim=-1)
            logprobs[slot] = rec_loss

            # KL divergence with latent prior
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_sum += kl.sum(dim=-1)

            # KL divergence between mask and reconstruction distributions
            mask_p = Bernoulli(probs=clamp_probs(log_mask))
            mask_q = Bernoulli(probs=clamp_probs(log_mask_rec))
            mask_kl = kl_divergence(mask_p, mask_q).view(batch_size, -1)
            mask_kl_sum += mask_kl.sum(dim=-1)

            recs[slot] = x_rec.detach()
            masks[slot] = log_mask.detach()

        r1 = -torch.logsumexp(logprobs, dim=0).mean()
        assert not torch.isnan(r1), 'nan in mse'
        r2 = kl_sum.mean()
        assert not torch.isnan(r2), 'nan in kl'
        r3 = mask_kl_sum.mean()
        assert not torch.isnan(r3), 'nan in mask kl'

        return r1, r2, r3, recs, masks
