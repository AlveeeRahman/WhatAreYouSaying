"""How much of Whisper's spill-over comes from its feature extractor and not from the encoder?

Whisper's log-mel input is floored at 8 below the loudest value of the clip. Erasing the degraded
stretch can lower that loudest value, which lowers the floor and changes the input in quiet frames
far from the stretch, including the silent padding up to 30 s. The spill-over outside the stretch
then mixes two effects. This check recomputes it with the floor held at the original clip's value,
so the input outside the erased stretch is the same as before and only the encoder can move the
clean frames. It runs the 60 controlled clips, where the degraded 2 s is known.

Needs the controlled set (experiments/controlled.py prepare). Writes experiments/results/checks/spillover_frontend.{json,md}.
Usage: python experiments/check_spillover_frontend.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from common import REVISIONS

from ways import load_backend
from ways.latent import degraded_frames, erase_frames
from ways.metrics import bootstrap_ci

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache/controlled"
OUT = HERE / "results/checks"
MODELS = {"whisper-tiny": "openai/whisper-tiny", "whisper-base": "openai/whisper-base"}
N_SAMPLES, N_FFT, HOP = 480000, 400, 160


def log_mel(be, wav):
    """Whisper's log10 mel power of one clip padded to 30 s, before the floor. -> [80, 3000]"""
    x = torch.zeros(N_SAMPLES, device=be.device); x[: len(wav)] = torch.as_tensor(wav, device=be.device)
    X = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT, device=be.device), return_complex=True)
    filters = torch.as_tensor(be.proc.feature_extractor.mel_filters, device=be.device, dtype=torch.float32)
    return torch.clamp(filters.T @ (X[:, :-1].abs() ** 2), min=1e-10).log10()


def features(L, top):
    return ((torch.maximum(L, top - 8.0) + 4.0) / 4.0)[None]


def states(be, feats, n20):
    with torch.no_grad():
        hs = be.model.model.encoder(feats, output_hidden_states=True).hidden_states
    return np.stack([h[0, :n20].float().cpu().numpy() for h in hs])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(HERE / "results/controlled/manifest.csv")
    S = dict(n_clips=int(len(man)), models={})
    for name, mid in MODELS.items():
        be = load_backend("hf-whisper", mid, revision=REVISIONS[mid])
        rows, worst = [], 0.0
        for r in man.itertuples():
            wav, _ = sf.read(CACHE / f"{r.id}_deg.wav", dtype="float32")
            n20 = len(wav) // 320
            d = degraded_frames(None, n20, 0.5, (r.win_start, r.win_end))
            L0, Le = log_mel(be, wav), log_mel(be, erase_frames(wav, d))
            f0 = features(L0, L0.max())
            worst = max(worst, float((f0 - be.features(wav[None])[0]).abs().max()))
            h0 = states(be, f0, n20)
            row = dict(id=r.id, degradation=r.degradation, floor_moved=float(L0.max() - Le.max()))
            for key, fe in (("own_floor", features(Le, Le.max())), ("fixed_floor", features(Le, L0.max()))):
                he = states(be, fe, n20)
                shift = np.linalg.norm(h0 - he, axis=2) / (np.linalg.norm(h0, axis=2) + 1e-9)
                row[key] = [float(s[~d].mean()) for s in shift]
                row[key + "_input_change_outside"] = float((f0 - fe)[0][:, np.repeat(~d, 2)[: 2 * n20].nonzero()[0]].abs().max())
            rows.append(row)
        own, fixed = np.array([r["own_floor"] for r in rows]), np.array([r["fixed_floor"] for r in rows])
        S["models"][name] = dict(max_feature_error_vs_extractor=worst, clips_where_floor_moved=int(sum(r["floor_moved"] > 1e-6 for r in rows)),
                                 layers=[dict(layer=l, own_floor=bootstrap_ci(own[:, l]), fixed_floor=bootstrap_ci(fixed[:, l]),
                                              difference=bootstrap_ci(own[:, l] - fixed[:, l])) for l in range(own.shape[1])])
        print(name, "feature error", worst, "floor moved in", S["models"][name]["clips_where_floor_moved"], "of", len(rows), flush=True)
    (OUT / "spillover_frontend.json").write_text(json.dumps(S, indent=1))
    md = []
    for name, m in S["models"].items():
        md += [f"**{name}: spill-over outside the erased 2 s, {S['n_clips']} controlled clips, floor moved in {m['clips_where_floor_moved']}**", "",
               "| layer | floor from the erased clip (as shipped) | floor held at the original | difference |", "|---|---|---|---|"]
        f = lambda c: f"{c['mean']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]"
        md += [f"| {l['layer']} | {f(l['own_floor'])} | {f(l['fixed_floor'])} | {f(l['difference'])} |" for l in m["layers"]] + [""]
    (OUT / "spillover_frontend.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
