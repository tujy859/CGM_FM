import torch
import numpy as np
from utils.glucodensity_utils import compute_glucodensity_patches_from_cgm

class DataTransformer:
    def __init__(self, config):
        self.config = config

    def transform(self, x):
        raise NotImplementedError

    def encode(self, encoder, x):
        raise NotImplementedError

class MaskedPatchDataTransformer(DataTransformer):
    '''
        @brief: M2 patchifier that carries the observation mask alongside the
                values (mask贯通) and exposes per-patch observation density.
                Missing cells must be encoded as 0.0 with mask=0 (normalized space).
    '''
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.get("patch_size", 12)
        self.mean = config.get("mean", None)
        self.std = config.get("std", None)

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    @staticmethod
    def patchify(x, patch_size):
        # x: (T,) -> (N, L)
        num_patches = len(x) // patch_size
        return x[: num_patches * patch_size].reshape(num_patches, patch_size)

    def transform(self, x, obs_mask=None):
        # x: (T,) values; obs_mask: (T,) {0,1}; returns dict of patches + density
        if self.mean is not None and self.std is not None:
            m = obs_mask if obs_mask is not None else np.ones_like(x, dtype=bool)
            x = x.copy()
            x[m.astype(bool)] = (x[m.astype(bool)] - self.mean) / self.std

        if obs_mask is None:
            obs_mask = np.ones_like(x, dtype=np.float32)

        x_p = self.patchify(np.asarray(x, dtype=np.float32), self.patch_size)
        m_p = self.patchify(np.asarray(obs_mask, dtype=np.float32), self.patch_size)
        density = m_p.mean(axis=1)  # (N,) observation density per patch

        return {
            "patches": torch.tensor(x_p).float(),
            "obs_mask": torch.tensor(m_p).float(),
            "density": torch.tensor(density).float(),
        }

    def encode(self, encoder, x, x_mark=None):
        # dual-channel input: values + observation mask, both (B, C, T)
        if isinstance(x, (tuple, list)):
            x = torch.cat([t.unsqueeze(1) for t in x], dim=1)
        emb, proj = encoder(x, x_mark)
        return torch.mean(emb, dim=1)

class PatchDataTransformer(DataTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.get("patch_size", 12)
        self.mean = config.get("mean", None)
        self.std = config.get("std", None)

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        # Normalize
        if self.mean is not None and self.std is not None:
            mask = x != -1
            x[mask] = (x[mask] - self.mean) / self.std
        
        # Patchify
        if len(x) % self.patch_size != 0:
            # add padding
            padding_size = self.patch_size - (len(x) % self.patch_size) 
            padding_x = np.full(padding_size, -1)
            x = np.concatenate([x, padding_x])
        
        num_patches = len(x) // self.patch_size
        x_patches = [x[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]
        x_patches_tensor = torch.tensor(np.array(x_patches)).float()
        return x_patches_tensor

    def encode(self, encoder, x, x_mark=None):
        emb, proj = encoder(x, x_mark)        
        return torch.mean(emb, dim=1)

class TokenDataTransformer(DataTransformer):
    def __init__(self, config):
        super().__init__(config)
        # From our csv CGM. num_bins must match the GluFormer checkpoint's
        # vocab_size (278 for the released v5 artifact); fall back to 280
        # (the current default) when not supplied.
        self.num_bins = config.get("num_bins") or 280
        self.min_glucose = 40
        self.max_glucose = 320 # based on max data and what can be measured by the device

    def glucose_to_bin(self, g):
        g = np.clip(g, self.min_glucose, self.max_glucose)
        width = (self.max_glucose - self.min_glucose) / self.num_bins
        bin_idx = np.floor((g - self.min_glucose) / width).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1) # clip bin idx greater than what we know
        return bin_idx

    def transform(self, x):
        x_tokenized = [self.glucose_to_bin(int(s)) for s in x]
        x_tokenized = np.array(x_tokenized)
        return x_tokenized

    def encode(self, encoder, x):
        device = next(encoder.parameters()).device
        x = x.to(device).long()
        encoded = encoder(x)
        return torch.mean(encoded, dim=1)

class FlatDataTransformer(DataTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.mean = config.get("mean", None)
        self.std = config.get("std", None)

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        # Normalize | for TSFM this will be None and skip normalization
        if self.mean is not None and self.std is not None:
            mask = x != -1
            x[mask] = (x[mask] - self.mean) / self.std
        return torch.tensor(x).float()

    def encode(self, encoder, x):
        # Flat input, just pass through.
        # Some encoders (e.g., JEPA-style or TS2Vec wrappers) return (emb, proj),
        # where we only want the embedding tensor.
        output = encoder(x)
        if isinstance(output, tuple):
            output = output[0]
        return output
