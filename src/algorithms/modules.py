import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import algorithms.vit.vit as vit
import math


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        assert size in {84, 96, 100}, f'unexpected size: {size}'
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, 'input must be a 4D tensor'
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
        if self.size == 96:
            p = 2
        elif self.size == 84:
            p = 8
        else:
            raise ValueError(f"unexpected size {self.size}")
        return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x/255.


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SODAMLP(nn.Module):
    def __init__(self, projection_dim, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)


class SharedTransformer(nn.Module):
    def __init__(self, obs_shape, patch_size=8, embed_dim=128, depth=4, num_heads=8, mlp_ratio=1., qvk_bias=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.frame_stack = obs_shape[0]//3
        self.img_size = obs_shape[-1]
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qvk_bias = qvk_bias

        self.preprocess = nn.Sequential(CenterCrop(size=self.img_size), NormalizeImg())
        self.transformer = vit.VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=self.frame_stack*3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qvk_bias,
        ).cuda()
        self.out_shape = _get_out_shape_cuda(obs_shape, nn.Sequential(self.preprocess, self.transformer))

    def forward(self, x):
        x = self.preprocess(x)
        return self.transformer(x)


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.img_size = obs_shape[-1]
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [CenterCrop(size=self.img_size), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, shared, head, projection):
        super().__init__()
        self.shared = shared
        self.head = head
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared(x)
        x = self.head(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class Actor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.mlp.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False, compute_attrib=False):
        x = self.encoder(x, detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(self.encoder.out_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.out_dim, action_shape[0], hidden_dim)

    def forward(self, x, action, detach=False):
        x = self.encoder(x, detach)
        return self.Q1(x, action), self.Q2(x, action)


class CURLHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class InverseDynamics(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

    def forward(self, x, x_next):
        h = self.encoder(x)
        h_next = self.encoder(x_next)
        joint_h = torch.cat([h, h_next], dim=1)
        return self.mlp(joint_h)


class SODAPredictor(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.mlp = SODAMLP(
            encoder.out_dim, hidden_dim, encoder.out_dim
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(self.encoder(x))


class AttributionDecoder(nn.Module):
    def __init__(self, action_shape, emb_dim=100):
        super().__init__()
        self.proj = nn.Linear(in_features=emb_dim+action_shape, out_features=14112)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, padding=1)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        x = self.proj(x).view(-1, 32, 21, 21)
        x = self.relu(x)
        x = self.conv1(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class AttributionPredictor(nn.Module):
    def __init__(self, action_shape, encoder, emb_dim=100):
        super().__init__()
        self.encoder = encoder
        self.decoder = AttributionDecoder(action_shape, encoder.out_dim)
        # self.features_decoder = nn.Sequential(
        #     nn.Linear(emb_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, emb_dim)
        # )

    def forward(self, x, action):
        x = self.encoder(x)
        return self.decoder(x, action)


class MaskerNet(nn.Module):
    def __init__(self, obs_shape, args):
        super().__init__()
        num_masks = args.frame_stack
        assert len(obs_shape) == 3
        self.img_size = obs_shape[-1]
        self.layers = [
            CenterCrop(size=self.img_size), NormalizeImg(),
            nn.Conv2d(obs_shape[0] // num_masks, args.masker_num_filters, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
        ]
        for _ in range(2, args.masker_num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(args.masker_num_filters, args.masker_num_filters, 3, stride=1, padding=1, padding_mode='zeros'))
        self.layers += [nn.ReLU(),
                        nn.Conv2d(args.masker_num_filters, 1, 3, stride=1, padding=1, padding_mode='zeros'),
                        nn.Sigmoid()]
        self.layers = nn.Sequential(*self.layers)
        in_shape = (obs_shape[0] // num_masks, obs_shape[1], obs_shape[2])
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)
        self.mask_type = args.mask_type
        self.threshold = args.mask_threshold
        self.threshold_type = args.mask_threshold_type
        self.current_soft_mask = None
        self._thresholds = []

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value):
        self._thresholds = value

    def forward(self, x, log_mask_stats=False, test_env=False):
        mask = self.layers(x)
        if log_mask_stats:
            self.current_soft_mask = mask
        if self.mask_type == 'hard' or self.mask_type == 'mixed' and test_env:
            if self.threshold_type == 'fix':
                mask = (mask > self.threshold).float()
            elif self.threshold_type == 'avg':
                mask = (mask > mask.mean()).float()
            elif self.threshold_type == 'quantile':
                threshold = torch.kthvalue(mask.view(-1), int((1 - self.threshold) * mask.numel()) + 1).values
                self._thresholds.append(threshold)
                mask = (mask > threshold).float()
            else:
                raise NotImplementedError
        return mask
