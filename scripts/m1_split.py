"""M1 语料划分：subject-disjoint 铁律。

评估队列（held-out）：
  cgmacros_dexcom 30 人（标签最全） / shanghait2dm 65 人 / hall_2018 57 人（glucotype 自算）
预训练池（共 412 人）：
  colas_2019 208 + cgm_jepa_pre 22(S前缀) + shanghait1dm 12 + shanghait2dm 35
  + cgmacros_dexcom 15 + bigideas 16 + bris 20 + uchtt1dm 20 + park_2025 38 + d1namo 9 + t1d_uom 17
  （CGMacros_Libre 不入池：与 Dexcom 同人同段，避免同人双份；HUPA-UCM/AZT1D 放弃下载）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Coding\Work\CGM_FM")
UNI = ROOT / "data/unified"
SEED = 42


def subjects_of(key):
    return sorted(pd.read_csv(UNI / f"{key}.csv", usecols=["subject"]).subject.unique())


def main():
    rng = np.random.RandomState(SEED)

    def split_pool(items, n_eval):
        items = list(items)
        rng.shuffle(items)
        return items[:n_eval], items[n_eval:]

    cg_e, cg_p = split_pool(subjects_of("cgmacros_dexcom"), 30)
    sh_e, sh_p = split_pool(subjects_of("shanghait2dm"), 65)

    pretrain = (
        subjects_of("colas_2019") + subjects_of("cgm_jepa_pre")
        + subjects_of("shanghait1dm") + sh_p + cg_p
        + subjects_of("bigideas") + subjects_of("bris_t1d_open")
        + subjects_of("uchtt1dm") + subjects_of("park_2025")
        + subjects_of("d1namo") + subjects_of("t1d_uom")
    )
    splits = {
        "seed": SEED,
        "pretrain": pretrain,
        "eval": {
            "cgmacros": cg_e,
            "shanghait2dm": sh_e,
            "hall": subjects_of("hall_2018"),
        },
        "notes": "cgmacros_libre excluded from pretraining (same subjects as dexcom); "
                 "hupa-ucm & azt1d dropped (Mendeley unreachable); "
                 "park_2025 subjects are per-rep segments sid#rN of 38 persons",
    }
    assert not (set(pretrain) & set(cg_e + sh_e + subjects_of("hall_2018"))), "leak!"

    out = ROOT / "data/splits.json"
    out.write_text(json.dumps(splits, indent=1), encoding="utf-8")
    print(f"pretrain: {len(pretrain)} subjects")
    for k, v in splits["eval"].items():
        print(f"eval {k}: {len(v)}")


if __name__ == "__main__":
    main()
