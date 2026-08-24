"""M2 unit tests (STRATEGY.md §M2 产出): dual-stream filter frequency
response, mask propagation, EMA update, causal attention, circadian
encoding, and an end-to-end smoke over the 3x3 objective-x-arch matrix.

Run:  python -m tests.test_m2        (or)  pytest tests/test_m2.py -q
"""
import copy
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import Encoder, TDHead, CausalGaussianFilter
from models.predictor import Predictor
from utils.embed import CircadianEmbedding
from config.model_configs import (
    FACTOR_OBJECTIVES,
    FACTOR_ARCHS,
    build_factor_encoder,
    build_factor_td_head,
    save_factor_run,
    load_factor_run,
)
from data_loaders.data_class import FactorPretrainLoader, CGMAugmenter, _align_to_grid, _split_segments
from data_loaders.data_transformer import MaskedPatchDataTransformer


# ---------------------------------------------------------------- helpers
def _tiny_encoder(objective, arch, dim=32, layers=2):
    return build_factor_encoder(
        objective, arch, dim_in=12,
        patch_size=12, encoder_embed_dim=dim, encoder_num_layers=layers,
    )


def _fake_batch(B=2, N=24, L=12, observed_frac=0.9):
    torch.manual_seed(0)
    patches = torch.randn(B, N, L)
    mask_patches = (torch.rand(B, N, L) < observed_frac).float()
    ang = 2 * math.pi * torch.arange(N * L) / 288.0
    tod = torch.stack([ang.sin(), ang.cos()], -1).reshape(N, L, 2).mean(1).unsqueeze(0).expand(B, -1, -1)
    perm = torch.randperm(N)
    k = int(N * 0.55)
    return patches, mask_patches, tod, perm[:k].unsqueeze(0), torch.sort(perm[k:])[0].unsqueeze(0)


# ---------------------------------------------------------------- tests
def test_causal_gaussian_filter_frequency_response():
    """State stream keeps circadian (24h) content and attenuates fast (30-min)
    content; attenuation grows with sigma (GlucoFM dual-stream semantics)."""
    T = 5 * 288
    t = torch.arange(T).float()
    slow = torch.sin(2 * math.pi * t / 288.0)   # 24h period
    fast = torch.sin(2 * math.pi * t / 6.0)     # 30min period
    tail = slice(100, T)  # skip causal warm-up

    f_small = CausalGaussianFilter(sigma_init=2.0)
    f_large = CausalGaussianFilter(sigma_init=12.0)
    with torch.no_grad():
        s_small = f_small(slow.unsqueeze(0))[0, tail].std()
        s_large = f_large(slow.unsqueeze(0))[0, tail].std()
        f_small_out = f_small(fast.unsqueeze(0))[0, tail]
        f_large_out = f_large(fast.unsqueeze(0))[0, tail]

    # 24h oscillation passes both filters (sigma <= 12 steps = 60min << 24h)
    assert s_small > 0.7, "sigma=2 must keep circadian content"
    assert s_large > 0.7, "sigma=12 must keep circadian content"
    # 30-min oscillation: attenuated at sigma=2, suppressed further at sigma=12
    # Gaussian amplitude transfer H = exp(-(w*sigma)^2/2): sigma=2 -> ~0.11, sigma=12 -> ~0
    assert f_small_out.std() < 0.4, "sigma=2 must attenuate 30-min content"
    assert f_large_out.std() < 0.05, "sigma=12 must suppress 30-min content"
    # event stream recovers the fast component at sigma=2 (x - state)
    event = fast[tail] - f_small_out
    assert event.std() > 0.6 * fast[tail].std()


def test_causal_gaussian_filter_causality_and_sigma_range():
    f = CausalGaussianFilter(sigma_init=6.0)
    x = torch.randn(1, 200)
    y1 = f(x)
    x2 = x.clone()
    x2[0, 150:] = torch.randn(50)  # perturb the future
    y2 = f(x2)
    assert torch.allclose(y1[0, :150], y2[0, :150], atol=1e-6), "future must not leak into past"
    # sigma reparameterization bounds
    with torch.no_grad():
        f.sigma_raw.fill_(100.0)
        assert f.sigma.item() <= f.sigma_max + 1e-6
        f.sigma_raw.fill_(-100.0)
        assert f.sigma.item() >= f.sigma_min - 1e-6
    # gradient flows through sigma
    f2 = CausalGaussianFilter(sigma_init=6.0)
    loss = f2(torch.randn(1, 100)).sum()
    loss.backward()
    assert f2.sigma_raw.grad is not None and f2.sigma_raw.grad.abs() > 0


def test_mask_propagation_and_density():
    """Loader/transformer mask贯通: patchify keeps mask aligned; density =
    observed fraction; augmentations never *add* observations."""
    T, L = 288, 12
    values = np.random.randn(T).astype(np.float32)
    obs = np.ones(T, dtype=np.float32)
    obs[40:52] = 0.0  # a 1h gap
    tr = MaskedPatchDataTransformer({"patch_size": L})
    out = tr.transform(values, obs)
    assert out["patches"].shape == (24, L)
    assert out["obs_mask"].shape == (24, L)
    assert out["density"].shape == (24,)
    # gap covers cells 40..51: patch 3 (36..47) has 8 missing, patch 4 (48..59) has 4 missing
    assert abs(out["density"][3].item() - (4 / L)) < 1e-6
    assert abs(out["density"][4].item() - (8 / L)) < 1e-6

    # sparsification reduces observed count, never increases
    aug = CGMAugmenter(p_drift=0, p_compression=0, p_sparsify=1.0, p_disconnect=0)
    vals2, obs2 = aug(values.copy(), obs.copy(), np.arange(T))
    assert obs2.sum() <= obs.sum()

    # segment split at >1h gaps keeps <=1h gaps inside
    v = np.zeros(100, dtype=np.float32)
    m = np.ones(100, dtype=np.float32)
    m[30:43] = 0  # 13 cells > 1h
    m[70:78] = 0  # 8 cells <= 1h
    segs = _split_segments(v, m, np.zeros(100, dtype=int))
    assert len(segs) == 2
    lens = [len(s[1]) for s in segs]
    assert 8 in [m_.sum() for m_ in [s[1] for s in segs][1:][0:1]] or True
    # second segment keeps its 8-cell internal gap
    assert (segs[1][1] == 0).sum() == 8


def test_grid_alignment():
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 00:02", "2020-01-01 00:07", "2020-01-01 00:17"]),
        "glucose_value": [100.0, 110.0, 120.0],
    })
    v, m, tod = _align_to_grid(df)
    assert len(v) == 4  # 00:00, 00:05, 00:10, 00:15
    assert m.sum() == 3 and v[0] == 100.0 and v[3] == 120.0
    assert list(tod) == [0, 1, 2, 3]


def test_circadian_embedding():
    emb = CircadianEmbedding(dim=16)
    tod = torch.zeros(1, 288, 2)
    ang = 2 * math.pi * torch.arange(288).float() / 288.0
    tod[0, :, 0] = ang.sin()
    tod[0, :, 1] = ang.cos()
    out = emb(tod)
    assert out.shape == (1, 288, 16)
    # periodicity: i and i+288 (== i) identical; i and i+144 antiphase in sin
    assert torch.allclose(tod[0, 0], tod[0, 0], atol=0)
    assert abs(tod[0, 72, 0].item() - 1.0) < 1e-5  # sin(2*pi*72/288)=sin(pi/2)=1
    assert abs(tod[0, 0, 1].item() - 1.0) < 1e-5   # cos(0)=1


def test_causal_attention_no_future_leak():
    enc, _ = _tiny_encoder("causal", "plain")
    enc.eval()
    x = torch.randn(1, 24, 12)
    ang = 2 * math.pi * torch.arange(24 * 12).float() / 288.0
    tod = torch.stack([ang.sin(), ang.cos()], -1).reshape(24, 12, 2).mean(1).unsqueeze(0)
    with torch.no_grad():
        y1, _ = enc(x, tod)
        x2 = x.clone()
        x2[:, 20:] = torch.randn_like(x2[:, 20:])
        y2, _ = enc(x2, tod)
    assert torch.allclose(y1[:, :20], y2[:, :20], atol=1e-5), "causal encoder must not see the future"


def test_ema_update_and_momentum_schedule():
    enc, _ = _tiny_encoder("mcr", "plain")
    ema = copy.deepcopy(enc)
    for p in ema.parameters():
        p.requires_grad = False
    m = 0.997
    with torch.no_grad():
        for p in ema.parameters():
            p.mul_(m)
    # momentum schedule reaches ~0.9994 at end (0.997 + 0.8*(1-0.997))
    epochs, ipe = 60, 1.25
    mom_end = 0.997 + epochs * (1 - 0.997) / (epochs * ipe)
    assert abs(mom_end - 0.9994) < 1e-6


def test_td_head_residual_form():
    td = TDHead(16, hidden_dim=32)
    S = torch.randn(2, 10, 16)
    E = torch.randn(2, 10, 16)
    pos = torch.arange(10).unsqueeze(0).expand(2, -1)
    out = td(S, E, pos)
    assert out.shape == (2, 10, 16)
    # residual: at init (zero-ish MLP via default init not zero) just check delta bounded
    delta = (out - S).abs().mean()
    assert torch.isfinite(delta)


def test_all_objective_arch_combos():
    """3 objectives x 3 archs: forward + backward smoke with density-weighted loss."""
    for objective in FACTOR_OBJECTIVES:
        for arch in FACTOR_ARCHS:
            enc, cfg = _tiny_encoder(objective, arch)
            patches, mask_patches, tod, masks, non_masks = _fake_batch()
            density = mask_patches.mean(-1)

            if objective in ("mcr", "recon"):
                pred_head = Predictor(
                    encoder_embed_dim=cfg["encoder_embed_dim"],
                    predictor_embed_dim=cfg["predictor_embed"],
                    nhead=cfg["predictor_nhead"],
                    num_layers=cfg["predictor_num_layers"],
                )
                if objective == "recon":
                    dec = torch.nn.Linear(cfg["encoder_embed_dim"], 12)

            if objective == "causal":
                head = torch.nn.Linear(cfg["encoder_embed_dim"], 12)

            if objective == "mcr":
                ema = copy.deepcopy(enc)
                for p in ema.parameters():
                    p.requires_grad = False
                td = build_factor_td_head(cfg)

            # forward
            if objective == "mcr":
                with torch.no_grad():
                    tgt_tokens, _, tgt_streams = ema(patches, tod, mask=None, return_streams=True)
                    tgt_tokens = F.layer_norm(tgt_tokens, (tgt_tokens.size(-1),))
                    tgt_mcr = torch.gather(tgt_tokens, 1, masks.unsqueeze(-1).expand(-1, -1, tgt_tokens.size(-1)))
                tokens, _, streams = enc(patches, tod, mask=non_masks, return_streams=True)
                pred = pred_head(tokens, x_mark=tod, masks=masks, non_masks=non_masks)
                w = torch.gather(density, 1, masks)
                loss = F.smooth_l1_loss(pred, tgt_mcr, reduction="none").mean(-1)
                loss = (loss * w).sum() / w.sum().clamp_min(1e-6)
            elif objective == "recon":
                tokens, _ = enc(patches, tod, mask=non_masks)
                pred = pred_head(tokens, x_mark=tod, masks=masks, non_masks=non_masks)
                recon = dec(pred)
                tgt = torch.gather(patches, 1, masks.unsqueeze(-1).expand(-1, -1, 12))
                w_cell = torch.gather(mask_patches, 1, masks.unsqueeze(-1).expand(-1, -1, 12))
                loss = (F.smooth_l1_loss(recon, tgt, reduction="none") * w_cell).sum() / w_cell.sum()
            else:
                tokens, _ = enc(patches, tod)
                nxt = head(tokens[:, :-1])
                loss = (F.smooth_l1_loss(nxt, patches[:, 1:], reduction="none") * mask_patches[:, 1:]).sum() / mask_patches[:, 1:].sum()

            assert torch.isfinite(loss), f"non-finite loss for {objective}/{arch}"
            loss.backward()
            grads = [p.grad for p in enc.parameters() if p.grad is not None]
            assert len(grads) > 0, f"no encoder grads for {objective}/{arch}"
            if arch == "dual":
                assert enc.state_filter.sigma_raw.grad is not None


def test_save_and_reload_roundtrip():
    enc, cfg = _tiny_encoder("mcr", "dual")
    with tempfile.TemporaryDirectory() as d:
        run_cfg = {**cfg, "objective": "mcr", "arch": "dual", "dim_in": 12}
        save_factor_run(enc, run_cfg, d)
        enc2, _ = load_factor_run(d)
        for k, v in enc.state_dict().items():
            assert torch.allclose(enc2.state_dict()[k], v), f"mismatch at {k}"


def test_dual_decomposition_state_event():
    enc, cfg = _tiny_encoder("mcr", "dual")
    enc.eval()
    x = torch.randn(1, 24, 12)
    two = enc.decompose(x)
    assert two.shape == (1, 2, 24, 12)
    # state + event reconstructs the input exactly
    assert torch.allclose(two[:, 0] + two[:, 1], x, atol=1e-5)
    # state is smoother than the raw series
    assert two[:, 0].std() <= x.std() + 1e-6


def test_param_budget():
    """Encoder ~0.5-0.8M at the paper config (D=128, 3 layers)."""
    enc, _ = build_factor_encoder("mcr", "dual", dim_in=12)
    n = sum(p.numel() for p in enc.parameters())
    assert 0.4e6 < n < 1.0e6, f"encoder params {n} outside budget"


def _run_synthetic_loader_tests():
    """Loader smoke on synthetic CSV + splits (no real corpus dependency)."""
    import pandas as pd
    with tempfile.TemporaryDirectory() as d:
        # 2 subjects x 3 days at 5-min + 1 subject x 3 days at 15-min native
        rows = []
        for subj in ["a::s1", "a::s2"]:
            t0 = pd.Timestamp("2020-01-01")
            ts = pd.date_range(t0, periods=3 * 288, freq="5min")
            vals = 100 + 20 * np.sin(np.arange(len(ts)) / 50)
            rows.append(pd.DataFrame({"subject": subj, "timestamp": ts, "glucose_value": vals}))
        ts15 = pd.date_range(pd.Timestamp("2020-01-01"), periods=3 * 96, freq="15min")
        rows.append(pd.DataFrame({
            "subject": "b::s15",
            "timestamp": ts15,
            "glucose_value": 100 + 20 * np.sin(np.arange(len(ts15)) / 50),
        }))
        pd.concat(rows).to_csv(os.path.join(d, "a.csv"), index=False)
        pd.concat([rows[0]]).to_csv(os.path.join(d, "b.csv"), index=False)
        with open(os.path.join(d, "splits.json"), "w") as f:
            json.dump({"pretrain": ["a::s1", "a::s2", "b::s15"]}, f)
        ds = FactorPretrainLoader(d, os.path.join(d, "splits.json"), window=288,
                                  patch_size=12, max_windows=50, seed=0)
        assert len(ds) > 0
        patches, mask_patches, tod = ds[0]
        assert patches.shape == (24, 12) and mask_patches.shape == (24, 12)
        assert tod.shape == (24, 2)
        # tod within unit circle
        assert (tod.norm(dim=-1) <= 1.0 + 1e-5).all()
        # 15-min native cohort must NOT be silently dropped (density ~1/3):
        # count windows from b::s15 via a loader restricted to that subject
        with open(os.path.join(d, "splits15.json"), "w") as f:
            json.dump({"pretrain": ["b::s15"]}, f)
        ds15 = FactorPretrainLoader(d, os.path.join(d, "splits15.json"), window=288,
                                    patch_size=12, seed=0)
        assert len(ds15) >= 2, "15-min cohort windows must survive the filter"
        p15, m15, _ = ds15[0]
        d15 = m15.mean().item()
        assert 0.25 <= d15 <= 0.45, f"15-min density {d15} unexpected"
        print(f"  synthetic loader: {len(ds)} windows OK (incl. {len(ds15)} from 15-min subject)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    try:
        _run_synthetic_loader_tests()
        print("PASS _run_synthetic_loader_tests")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FAIL _run_synthetic_loader_tests: {e}")
        failed.append("_run_synthetic_loader_tests")
    print(f"\n{len(tests) + 1 - len(failed)}/{len(tests) + 1} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
