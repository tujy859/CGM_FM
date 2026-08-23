import torch
import torch.nn as nn
import numpy as np

from huggingface_hub import PyTorchModelHubMixin

from utils.embed import DataEmbedding
from utils.modules import *
from utils.mask_utils import *

ARCHS = ("plain", "dual", "cnn")


class CausalGaussianFilter(nn.Module):
    '''
        @brief: Learnable causal Gaussian smoothing for dual-stream state/event
                decomposition (GlucoFM). Sigma is reparameterized with a sigmoid
                into [sigma_min, sigma_max] grid steps (5-min units), so it stays
                in a valid range while receiving gradients.
    '''
    def __init__(self, sigma_init=6.0, sigma_min=2.0, sigma_max=12.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # init raw so that sigmoid(raw) maps to sigma_init
        target = (sigma_init - sigma_min) / (sigma_max - sigma_min)
        target = min(max(target, 1e-4), 1 - 1e-4)
        self.sigma_raw = nn.Parameter(torch.tensor(float(np.log(target / (1.0 - target)))))

        self.kernel_size = int(3 * sigma_max) + 1  # cover 3*sigma_max causal taps

    @property
    def sigma(self):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma_raw)

    def _kernel(self, device, dtype):
        idx = torch.arange(self.kernel_size, device=device, dtype=dtype)
        var = self.sigma.to(dtype) ** 2
        k = torch.exp(-(idx ** 2) / (2.0 * var))
        return k / k.sum()

    def forward(self, x):
        # x: (B, T) or (B, C, T) -> causal smoothed version, same shape
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        B, C, T = x.shape
        k = self._kernel(x.device, x.dtype)  # (K,)
        # left-pad so output[i] depends only on x[<=i]
        xp = torch.nn.functional.pad(x, (self.kernel_size - 1, 0))
        weight = k.view(1, 1, 1, -1).expand(1, C, 1, -1).reshape(C, 1, self.kernel_size)
        out = torch.nn.functional.conv1d(xp, weight, groups=C)
        return out.squeeze(1) if squeeze else out


class TDHead(nn.Module):
    '''
        @brief: Temporal-dynamics head (GlucoFM TD objective): residual
                next-patch state prediction,
                    S_next = S + g(S, E, tau)
                where tau is a learnable per-position time embedding.
    '''
    def __init__(self, dim, hidden_dim=256, max_len=5000):
        super().__init__()
        self.tau_embed = nn.Embedding(max_len, dim)
        self.g = MLP(in_features=3 * dim, hidden_features=hidden_dim, out_features=dim)

    def forward(self, S, E, pos_idx):
        # S, E: (B, N, D) token sequences, or (P, D) flat pair samples
        # pos_idx: (B, N) | (N,) | (P,) absolute patch positions within the window
        squeeze = S.dim() == 2
        if squeeze:
            S = S.unsqueeze(1)
            E = E.unsqueeze(1)
            pos_idx = pos_idx.reshape(-1, 1)
        if pos_idx.dim() == 1:
            pos_idx = pos_idx.unsqueeze(0).expand(S.size(0), -1)
        tau = self.tau_embed(pos_idx)                      # (B, N, D)
        out = S + self.g(torch.cat([S, E, tau], dim=-1))   # S_next prediction
        return out.squeeze(1) if squeeze else out


class ConvBackbone(nn.Module):
    '''
        @brief: A3 CNN backbone (PatchTST-style hierarchical temporal convolutions,
                no attention). Causally dilated residual blocks keep it usable for
                the causal objective; dilation doubles per block.
    '''
    def __init__(self, dim, num_layers=3, kernel_size=3, drop=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = [2 ** i for i in range(num_layers)]
        self.pads = [(kernel_size - 1) * d for d in self.dilations]  # causal (left-only) padding
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel_size, dilation=d, padding=0) for d in self.dilations
        )
        self.gate_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.gate_convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel_size, dilation=d, padding=0) for d in self.dilations
        )
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, N, D) -> (B, N, D)
        h = x.transpose(1, 2)  # (B, D, N)
        for i in range(len(self.convs)):
            pad = self.pads[i]
            res = h
            v = self.norms[i](h.transpose(1, 2)).transpose(1, 2)
            v = self.convs[i](torch.nn.functional.pad(v, (pad, 0)))
            g = self.gate_norms[i](h.transpose(1, 2)).transpose(1, 2)
            g = self.gate_convs[i](torch.nn.functional.pad(g, (pad, 0)))
            h = res + self.drop(self.act(v) * torch.sigmoid(g))
        return h.transpose(1, 2)


class Encoder(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/CRUISEResearchGroup/CGM-JEPA",
    pipeline_tag="feature-extraction",
    license="mit",
    tags=["cgm", "jepa", "self-supervised-learning", "time-series", "biosignal"],
):
    '''
        @brief: Encode input into latent representations. Used for
        both input and target encoder.

        Loadable via ``Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA",
        subfolder="cgm_jepa")`` (or ``subfolder="x_cgm_jepa")``

        M2 extension: ``arch`` in {"plain", "dual", "cnn"} factor matrix and
        ``causal`` attention for the causal objective. arch="dual" decomposes
        the raw series with a learnable CausalGaussianFilter into state/event
        streams before patch embedding.
    '''
    def __init__(
        self,
        dim_in,
        kernel_size,
        embed_dim,
        embed_bias,
        nhead,
        num_layers,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        jepa=False,
        embed_activation=nn.GELU(),
        time_inp_dim=5, # depend on freq map | 't' = 5
        arch="plain",
        causal=False,
        use_circadian=True,
        sigma_init=6.0,
        sigma_min=2.0,
        sigma_max=12.0,
    ):
        super().__init__()
        assert arch in ARCHS, f"arch must be one of {ARCHS}, got {arch}"
        self.arch = arch
        self.causal = causal
        self.use_circadian = use_circadian

        self.embed_dim = embed_dim
        self.activation = embed_activation if embed_activation else nn.GELU()

        # learnable causal Gaussian filter for the dual-stream architecture
        self.state_filter = CausalGaussianFilter(sigma_init, sigma_min, sigma_max) if arch == "dual" else None

        self.data_embedding = DataEmbedding(
            dim=embed_dim,
            in_channels=dim_in,
            patch_size=kernel_size,
            time_inp_dim=time_inp_dim,
            dropout=drop_rate,
            arch=arch,
            use_circadian=use_circadian,
        )

        if arch == "cnn":
            self.predictor_blocks = ConvBackbone(
                dim=embed_dim, num_layers=num_layers, drop=drop_rate
            )
        else:
            self.predictor_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=nhead,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        act_layer=nn.GELU,
                        norm_layer=norm_layer,
                    )
                    for i in range(num_layers)
                ]
            )

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.jepa = jepa

        # post-hoc stream projections for the TD head (dual arch);
        # plain/cnn reuse the token sequence for both streams
        if arch == "dual":
            self.state_proj = nn.Linear(embed_dim, embed_dim)
            self.event_proj = nn.Linear(embed_dim, embed_dim)

        self.proj = MLP(
            in_features=embed_dim,
            hidden_features=1024,
            out_features=48,
            act_layer=nn.GELU
        )

    def decompose(self, x):
        # x: (B, N, L) value patches -> (B, 2, N, L) state/event streams
        B, N, L = x.shape
        series = x.reshape(B, N * L)
        state = self.state_filter(series)
        event = series - state
        two = torch.stack([state, event], dim=1)          # (B, 2, T)
        return two.view(B, 2, N, L)

    def forward(self, x, x_mark=None, mask=None, return_streams=False):
        # plain/cnn: x (B, N, L) value patches
        # dual: x (B, N, L) raw patches (decomposed internally)
        # x_mark: (B, N, 2) circadian phase | (B, N, L, d_inp) legacy | None

        # optional timestamp
        if x_mark is not None:
            x_mark = x_mark.to(x.device)

        if x_mark is None:
            x_mark = torch.zeros(x.size(0), x.size(1), 2, device=x.device)

        if self.arch == "dual":
            x = self.decompose(x)

        x = self.data_embedding(x, x_mark) # (B, N, D)

        # apply mask. in the encoder, we only keep unmasked part
        if mask is not None and self.jepa:
            x = apply_mask(x, mask)

        # encode
        if self.arch == "cnn":
            x = self.predictor_blocks(x)
        else:
            for blk in self.predictor_blocks:
                x = blk(x, causal=self.causal)

        x = self.encoder_norm(x)

        if return_streams:
            if self.arch == "dual":
                state = self.state_proj(x)
                event = self.event_proj(x)
            else:
                state, event = x, x
            return x, self.proj(x), {"state": state, "event": event}

        return x, self.proj(x)
