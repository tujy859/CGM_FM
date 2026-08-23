import pandas as pd
import torch
import random
import numpy as np

import os
import json
import glob

from torch.utils.data import Dataset

from utils.timefeatures import time_features
from .base_loader.base_loader import CSVDataLoader, JSONDataLoader


def _align_to_grid(df, grid_minutes=5):
    """Align one subject's (timestamp, glucose) rows to a regular 5-min grid.

    Returns (values, obs_mask, tod_idx) numpy arrays over the full reindexed
    range. Missing grid cells get value 0.0 and mask 0. tod_idx is the grid
    step index within the day (0..287) used for circular encoding.
    """
    ts = pd.to_datetime(df["timestamp"])
    vals = pd.to_numeric(df["glucose_value"], errors="coerce")
    grid = ts.dt.floor(f"{grid_minutes}min")
    s = pd.Series(vals.values, index=grid)
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq=f"{grid_minutes}min")
    s = s.reindex(full_idx)

    obs_mask = (~s.isna()).to_numpy(dtype=np.float32)
    values = s.fillna(0.0).to_numpy(dtype=np.float32)
    minutes_of_day = full_idx.hour * 60 + full_idx.minute
    tod_idx = (minutes_of_day // grid_minutes).to_numpy()  # 0..287
    return values, obs_mask, tod_idx


def _split_segments(values, obs_mask, tod_idx, max_gap_cells=12):
    """Split a grid-aligned series at observation gaps longer than max_gap_cells
    (12 cells = 1h at 5-min). Shorter gaps stay inside the segment as mask=0."""
    missing = obs_mask == 0
    segs = []
    start = 0
    i = 0
    while i < len(missing):
        if missing[i]:
            j = i
            while j < len(missing) and missing[j]:
                j += 1
            if j - i > max_gap_cells:
                if i - start >= max_gap_cells + 1:
                    segs.append((start, i))
                start = j
            i = j
        else:
            i += 1
    if len(missing) - start >= max_gap_cells + 1:
        segs.append((start, len(missing)))
    return [
        (values[a:b], obs_mask[a:b], tod_idx[a:b]) for a, b in segs
    ]


class CGMAugmenter:
    """GlucoFM-style CGM-aware augmentations (STRATEGY.md §3).

    Operates on a normalized window (values, obs_mask, tod_idx) with raw-space
    mean/std supplied for multiplicative artifacts. All ops respect the
    observation mask (only observed cells carry signal).
    """
    def __init__(
        self,
        p_drift=0.25,
        p_compression=0.10,
        p_sparsify=0.40,
        p_disconnect=0.05,
        mean=0.0,
        std=1.0,
    ):
        self.p_drift = p_drift
        self.p_compression = p_compression
        self.p_sparsify = p_sparsify
        self.p_disconnect = p_disconnect
        self.mean = mean
        self.std = std

    def __call__(self, values, obs_mask, tod_idx):
        T = len(values)
        values = values.copy()

        # 1. baseline drift: slow sinusoid added to observed cells
        if random.random() < self.p_drift:
            amp = random.uniform(3.0, 15.0) / self.std  # raw mg/dL -> normalized
            period = random.uniform(6 * 12, 24 * 12)  # cells (6h..24h)
            phase = random.uniform(0, 2 * np.pi)
            t = np.arange(T, dtype=np.float32)
            values += (amp * np.sin(2 * np.pi * t / period + phase)).astype(np.float32)

        # 2. compression sudden drop: multiplicative raw-space artifact
        if random.random() < self.p_compression:
            length = random.randint(6, 24)  # 30min..2h
            start = random.randint(0, max(0, T - length))
            scale = random.uniform(0.6, 0.85)
            raw = values[start:start + length] * self.std + self.mean
            values[start:start + length] = ((raw * scale) - self.mean) / self.std

        # 3. structural sparsification: thin 5-min observations to 15-min
        if random.random() < self.p_sparsify:
            offset = random.randint(0, 2)
            keep = (np.arange(T) + offset) % 3 == 0
            obs_mask = obs_mask * keep.astype(np.float32)

        # 4. disconnection blocks: contiguous loss of signal
        if random.random() < self.p_disconnect:
            length = random.randint(12, 36)  # 1h..3h
            start = random.randint(0, max(0, T - length))
            obs_mask = obs_mask.copy()
            obs_mask[start:start + length] = 0.0

        return values, obs_mask


class FactorPretrainLoader(Dataset):
    '''M2 pretraining loader over the M1 unified corpus (data/unified/*.csv).

    - selects the subject-disjoint pretraining pool from data/splits.json
    - aligns each subject-segment to a 5-min grid with an observation mask
      (no interpolation; >1h gaps split segments, <=1h stay mask=0)
    - emits 24h windows; CGM-aware augmentations; mask ratio ~ U[mmin, mmax]
    - __getitem__ returns (patches, obs_mask_patches, tod_patches,
      mask_indices, non_mask_indices); density is recomputed by the caller
      from obs_mask_patches
    '''
    def __init__(
        self,
        data_dir,
        splits_path,
        window=288,
        stride=None,
        patch_size=12,
        mask_ratio_range=(0.5, 0.6),
        augment=True,
        min_obs_frac=0.5,
        max_windows=None,
        seed=43,
        grid_minutes=5,
        split_key="pretrain",
    ):
        super().__init__()
        self.window = window
        self.stride = stride if stride is not None else window
        self.patch_size = patch_size
        self.mask_ratio_range = mask_ratio_range
        self.augment = augment
        self.grid_minutes = grid_minutes

        with open(splits_path, "r") as f:
            splits = json.load(f)
        keep = set(splits[split_key])

        # read every unified CSV, keep only pool subjects
        subjects = {}
        for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
            name = os.path.basename(path)
            if name == "summary.csv":
                continue
            df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
            for subj, g in df.groupby("subject", sort=False):
                if subj in keep:
                    subjects[subj] = g[["timestamp", "glucose_value"]]

        missing = keep - set(subjects.keys())
        if missing:
            raise ValueError(f"{len(missing)} split subjects missing from corpus, e.g. {sorted(missing)[:3]}")

        # global stats over observed raw values (pre-normalization)
        all_obs = np.concatenate([g["glucose_value"].to_numpy(dtype=np.float64) for g in subjects.values()])
        self.stats = {"mean": float(all_obs.mean()), "std": float(all_obs.std())}

        # grid-align, split at >1h gaps, collect windows
        rng = np.random.default_rng(seed)
        self.windows = []  # (values, obs_mask, tod_idx)
        for subj in sorted(subjects.keys()):
            g = subjects[subj].sort_values("timestamp")
            values, obs_mask, tod_idx = _align_to_grid(g, grid_minutes)
            for seg_v, seg_m, seg_t in _split_segments(values, obs_mask, tod_idx):
                for start in range(0, len(seg_v) - self.window + 1, self.stride):
                    w_v = seg_v[start:start + self.window]
                    w_m = seg_m[start:start + self.window]
                    if w_m.mean() < min_obs_frac:
                        continue
                    self.windows.append((w_v, w_m, seg_t[start:start + self.window]))

        if max_windows is not None and len(self.windows) > max_windows:
            idx = rng.choice(len(self.windows), size=max_windows, replace=False)
            self.windows = [self.windows[i] for i in sorted(idx)]

        self.augmenter = CGMAugmenter(mean=self.stats["mean"], std=self.stats["std"])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        values, obs_mask, tod_idx = self.windows[idx]

        # normalize observed cells; missing stay 0.0 (= mean) with mask 0
        v = np.where(obs_mask > 0, (values - self.stats["mean"]) / self.stats["std"], 0.0).astype(np.float32)

        if self.augment:
            v, obs_mask = self.augmenter(v, obs_mask, tod_idx)

        # patchify values / mask; circular tod per patch
        N = self.window // self.patch_size
        L = self.patch_size
        patches = torch.tensor(v.reshape(N, L))
        mask_patches = torch.tensor(obs_mask.reshape(N, L))
        ang = 2 * np.pi * tod_idx.astype(np.float32) / 288.0
        tod = np.stack([np.sin(ang), np.cos(ang)], axis=-1)  # (T, 2)
        tod_patches = torch.tensor(tod.reshape(N, L, 2).mean(axis=1))  # (N, 2)

        # NOTE: mask ratio is sampled per BATCH in the training loop (collate
        # requires equal sizes); the ratio range stays U[mask_min, mask_max]
        # per STRATEGY.md §3, with per-sample independent permutations.
        return patches, mask_patches, tod_patches


class JEPALoader(CSVDataLoader):
    def __init__(self, *args, is_precompute_gluco=False, gluco_cache_path=None, **kwargs):
        """
        JEPA DataLoader with optional pre-computed glucodensity patches.
        
        Args:
            gluco_cache_path: Path to pre-computed glucodensity patches pickle file
            **kwargs: Other arguments passed to CSVDataLoader
        """
        super().__init__(*args, **kwargs)
        self.gluco_cache_path = gluco_cache_path
        self.gluco_cache = None
        self.is_precompute_gluco = is_precompute_gluco
        
        # Load pre-computed cache if provided
        if gluco_cache_path and os.path.exists(gluco_cache_path):
            import pickle
            print(f"Loading pre-computed glucodensity patches from {gluco_cache_path}...")
            with open(gluco_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.gluco_cache = cache_data['gluco_patches']
                print(f"✅ Loaded {len(self.gluco_cache)} pre-computed samples")
        elif gluco_cache_path:
            print(f"⚠️  Warning: Cache path provided but file not found: {gluco_cache_path}")
    
    def __getitem__(self, idx):
        subject, start_idx = self.samples[idx]

        # get subject's data
        subject_df = self.df[self.df[self.subject_col] == subject]
        ts_raw = subject_df[self.glucose_value_col].values

        # Skip normalization if (a) precomputing glucodensity, or (b) normalize flag explicitly off
        if self.is_precompute_gluco or not getattr(self, "normalize", True):
            ts = ts_raw
        else:
            assert self.normalize and self.global_mean is not None and self.global_std is not None, \
                "ERROR: No normalization done to load this sample"
            ts = (ts_raw - self.global_mean) / self.global_std

        ts = torch.tensor(ts).float()
        df_stamp = subject_df[[self.timestamp_col]]
        df_stamp["timestamp"] = pd.to_datetime(df_stamp.timestamp)
        timestamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        timestamp = torch.tensor(timestamp).float()

        # extract the window using the precomputed start_idx
        end_idx = start_idx + self.series_split_size
        selected_series = ts[start_idx:end_idx]
        selected_timestamp = timestamp[start_idx:end_idx]

        if len(selected_series) < self.series_split_size:
            # add padding
            padding_size = self.series_split_size - len(selected_series)
            padding_glucose = np.full(padding_size, selected_series[-1])

            # create timestamp padding by adding 5 minutes interval
            last_timestamp = pd.Timestamp(selected_timestamp[-1])
            padding_timestamp = pd.date_range(
                start=last_timestamp + pd.Timedelta(minutes=5),
                periods=padding_size,
                freq='5min'
            )

            selected_series = np.concatenate([selected_series, padding_glucose])
            selected_timestamp = np.concatenate([selected_timestamp, padding_timestamp])

        assert selected_series.shape[0] == selected_timestamp.shape[0], "Size between glucose and timestamp don't match"
        assert selected_series.shape[0] == self.series_split_size, f"Expected {self.series_split_size} but got {selected_series.shape[0]}"
        assert selected_timestamp.shape[0] == self.series_split_size, f"Expected {self.series_split_size} but got {selected_timestamp.shape[0]}"

        # divide the selected smaller time series into patches
        num_patches = len(selected_series) // self.patch_size
        patches = [selected_series[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]
        timestamp_patches = [selected_timestamp[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]

        # convert patches to tensor
        patches_tensor = torch.stack(patches)
        timestamp_patches_tensor = torch.stack(timestamp_patches)

        # create the mask for the patches
        num_masked_patches = int(num_patches * self.mask_ratio)
        mask_indices = random.sample(range(num_patches), num_masked_patches) if num_masked_patches > 0 else []
        non_mask_indices = [i for i in range(num_patches) if i not in mask_indices]

        mask_indices = torch.tensor(mask_indices, dtype=torch.long) if len(mask_indices) > 0 else torch.empty(0, dtype=torch.long)
        non_mask_indices = torch.tensor(non_mask_indices, dtype=torch.long) if len(non_mask_indices) > 0 else torch.empty(0, dtype=torch.long)

        if not self.use_time_feature:
            time_feat = torch.zeros_like(timestamp_patches_tensor)
        else:
            time_feat = timestamp_patches_tensor

        # Load pre-computed glucodensity patches if available; else compute on-the-fly
        if self.gluco_cache is not None:
            subject, start_idx = self.samples[idx]
            try:
                gluco_patches = self.gluco_cache[(subject, start_idx)]
                gluco_patches = torch.tensor(gluco_patches, dtype=torch.float32)
                return patches_tensor, time_feat, mask_indices, non_mask_indices, gluco_patches
            except KeyError:
                # Cache was built for different samples (e.g. different CSV/split); compute from patches
                from utils.glucodensity_utils import compute_glucodensity_patches_from_cgm
                p = getattr(self, "gluco_spatial_patch_size", 8)
                g = getattr(self, "gluco_gridsize", 32)
                gluco_np = compute_glucodensity_patches_from_cgm(
                    patches_tensor.numpy(), patch_size=p, gridsize=g
                )
                gluco_patches = torch.tensor(gluco_np, dtype=torch.float32)
                return patches_tensor, time_feat, mask_indices, non_mask_indices, gluco_patches
        else:
            return patches_tensor, time_feat, mask_indices, non_mask_indices

class GluFormerDataLoader(CSVDataLoader):
    def glucose_to_bin(self, g):
        g = np.clip(g, self.min_glucose, self.max_glucose)
        width = (self.max_glucose - self.min_glucose) / self.num_bins

        bin_idx = np.floor((g - self.min_glucose) / width).astype(int) # min = 40, g = 60 -> 20
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1) # will use the num_bins-th for PAD
        return bin_idx

    def __getitem__(self, idx):
        subject, start_idx = self.samples[idx]

        # assume we already have the vocab_size
        subject_df = self.df[self.df[self.subject_col] == subject]
        ts = subject_df[self.glucose_value_col].values

        # no normalization, we take the value as token like a word
        end_idx = start_idx + self.series_split_size
        selected_series = ts[start_idx:end_idx]

        # turn into bin token
        selected_series = [self.glucose_to_bin(int(s)) for s in selected_series]

        selected_series = np.array(selected_series)
        
        if len(selected_series) < self.series_split_size:
            # add padding
            padding_size = self.series_split_size - len(selected_series)
            padding_glucose = np.full(padding_size, self.num_bins) # last token is padding

            selected_series = np.concatenate([selected_series, padding_glucose])

        assert selected_series.shape[0] == self.series_split_size, f"Expected {self.series_split_size} but got {selected_series.shape[0]}"
        
        return selected_series
        
class ClassificationDataLoader(JSONDataLoader):
    def _extract_target(self, v):
        return v['y'][self.metabolic]['class']

    def _format_target(self, y):
        return torch.tensor(y, dtype=torch.long)
