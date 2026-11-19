"""Draw the 100-clip comparison sample from the saved, reproducible test slice.

Rule: simple random sample without replacement, numpy default_rng(SEED).choice(1500, 100), from the
1500-row test.csv of the pipeline repository (sha256 recorded below).
The test slice was fixed before any model was run and is disjoint from the development and load
slices; this script asserts that and hashes everything it read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import pipeline_repo
from ways.provenance import public_path, sha256

SEED = 8001
N = 100
HERE = Path(__file__).resolve().parent
PIPE = pipeline_repo(sys.argv[1] if len(sys.argv) > 1 else None)
PE = PIPE / "."

test = pd.read_csv(PE / "test.csv", usecols=["file", "label", "original_index", "duration"])
dev = pd.read_csv(PE / "development.csv", usecols=["file"])
load = pd.read_csv(PE / "heldout.csv", usecols=["file"])
assert len(test) == 1500, len(test)
assert not set(test.file) & set(dev.file) and not set(test.file) & set(load.file), "slices overlap"
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(test), N, replace=False))
s = test.iloc[idx].reset_index(drop=True)
s.to_csv(HERE / "sample100.csv", index=False)
prov = dict(rule=f"numpy.random.default_rng({SEED}).choice(len(test), {N}, replace=False), rows sorted by position",
            seed=SEED, n=N, test_csv=public_path(PE / "test.csv"), test_csv_sha256=sha256(PE / "test.csv"),
            test_rows=int(len(test)), development_rows=int(len(dev)), heldout_rows=int(len(load)),
            heldout_audio_read=False, sample_sha256=sha256(HERE / "sample100.csv"),
            labels=s.label.value_counts().to_dict(), total_duration_s=float(s.duration.sum()),
            clips=s.file.tolist())
(HERE / "sample100_provenance.json").write_text(json.dumps(prov, indent=1))
print(json.dumps({k: v for k, v in prov.items() if k != "clips"}, indent=1))
