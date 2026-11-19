"""Tables and figures over the 100-clip runs: per model WER and attribution summaries with bootstrap
CIs, paired model differences, the noise-focus ratio under each of the three AADD detectors, and
the comparison figure. Writes experiments/results/comparison.json and comparison.md."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ways.attribution import Attribution
from ways.erasure import band_edges_hz
from ways.metrics import bootstrap_ci, paired_bootstrap, token_mass
from ways.plots import comparison_figure

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE.parent / "figures"
MODELS = ["whisper-tiny", "whisper-base", "moonshine-base"]
DETS = {"v3": 0.02, "v3-lite": 0.02, "v2": 1.0}
sample = pd.read_csv(HERE / "sample100.csv")
clips = {m: pd.read_csv(RES / m / "clips.csv").set_index("clip").loc[sample.file].reset_index() for m in MODELS}
tokens = {m: pd.read_csv(RES / m / "tokens.csv") for m in MODELS}
edges = band_edges_hz(8)


def focus_under(det: str, m: str) -> np.ndarray:
    """Clip-level mean noise-focus ratio of model m under detector det (frames flagged at p >= 0.5
    for any class; for v2 the 'None' column is excluded from the any-class rule)."""
    res_s = DETS[det]
    out = []
    for clip in sample.file:
        att = Attribution.load(RES / m / "npz" / f"{clip}.npz")
        P = np.load(RES / "aadd" / det / f"{clip}.npy")
        if det == "v2":
            P = P[:, :-1]                      # drop 'None'
        flag = (P >= 0.5).any(1).astype(np.float32)
        per = att.patch_s / res_s
        f = np.zeros(att.n_t, np.float32)
        for t in range(att.n_t):
            lo, hi = int(np.floor(t * per)), int(np.ceil((t + 1) * per))
            seg = flag[lo:max(hi, lo + 1)]
            f[t] = seg.mean() if len(seg) else 0.0
        fbar = f.mean()
        if fbar <= 0 or fbar >= 1:
            out.append(np.nan); continue      # ratio undefined when nothing / everything is flagged
        mass = token_mass(att)
        prof = np.nansum(mass, axis=1)        # [K, T]
        ok = ~np.isnan(mass).any((1, 2))
        if not ok.any():
            out.append(np.nan); continue
        fr = (prof[ok] * f).sum(1) / fbar
        out.append(float(fr.mean()))
    return np.array(out)


S = dict(sample=json.loads((HERE / "sample100_provenance.json").read_text()), models={}, paired={}, detectors={})
for m in MODELS:
    c, t = clips[m], tokens[m]
    S["models"][m] = dict(
        n_clips=int(len(c)), n_tokens=int(len(t)), mean_duration=float(c.duration.mean()),
        seconds_per_clip=float(c.seconds.mean()), patches_per_clip=float(c.n_patches.mean()),
        wer=bootstrap_ci(c.wer), wer_corpus=float((c.wer_S + c.wer_D + c.wer_I).sum() / c.wer_N.sum()),
        surprisal0=bootstrap_ci(c.mean_surprisal0),
        top_share=bootstrap_ci(c.top_share_median),
        flip_any=bootstrap_ci(c.flip_any_frac),
        focus_ratio=bootstrap_ci(c.focus_ratio_mean),
        focus_ratio_informative=bootstrap_ci(c.loc[(c.frac_flagged > 0) & (c.frac_flagged < 1), "focus_ratio_mean"]),
        frac_flagged=bootstrap_ci(c.frac_flagged),
        n_all_flagged=int((c.frac_flagged >= 1).sum()), n_none_flagged=int((c.frac_flagged <= 0).sum()),
        band_shares=[float(t[f"band{b}"].mean()) for b in range(8)],
        band_edges_hz=[round(float(e)) for e in edges],
    )
    if "dist_median_s" in c:
        S["models"][m]["dist_median_s"] = bootstrap_ci(c.dist_median_s)
        S["models"][m]["within_0p3"] = bootstrap_ci(c.within_0p3)
for a, b in [("whisper-base", "moonshine-base"), ("whisper-tiny", "whisper-base"), ("whisper-tiny", "moonshine-base")]:
    S["paired"][f"{a} - {b}"] = dict(wer=paired_bootstrap(clips[a].wer, clips[b].wer),
                                     focus_ratio=paired_bootstrap(clips[a].focus_ratio_mean, clips[b].focus_ratio_mean),
                                     top_share=paired_bootstrap(clips[a].top_share_median, clips[b].top_share_median))
# the same question under three detectors
for det in DETS:
    S["detectors"][det] = {}
    for m in MODELS:
        fr = focus_under(det, m)
        S["detectors"][det][m] = bootstrap_ci(fr)
    fr_t, fr_b, fr_m = (focus_under(det, m) for m in MODELS)
    S["detectors"][det]["paired whisper-base - moonshine-base"] = paired_bootstrap(fr_b, fr_m)
info = json.loads((RES / "aadd" / "info.json").read_text())
S["detector_info"] = {k: dict(classes=v["classes"], resolution_s=v["resolution_s"], n_params=v["n_params"]) for k, v in info.items() if k in DETS}
S["detector_info"]["v3_vs_pipeline_csv"] = info["v3_vs_pipeline_csv"]
(RES / "comparison.json").write_text(json.dumps(S, indent=1, default=float))
comparison_figure(clips, tokens, FIG / "comparison_100clips.png", title="100 random test clips (seed 8001), three small models", band_edges=edges)


def ci(d, f="{:.2f}"):
    return f"{f.format(d['mean'])} [{f.format(d['ci'][0])}, {f.format(d['ci'][1])}]"


lines = ["| model | tokens | WER (mean over clips) | corpus WER | top-patch share | tokens with any flip | noise-focus ratio (v3) | focus, informative clips |", "|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    d = S["models"][m]
    lines.append(f"| {m} | {d['n_tokens']} | {ci(d['wer'])} | {d['wer_corpus']:.3f} | {ci(d['top_share'])} | {ci(d['flip_any'])} | {ci(d['focus_ratio'])} | {ci(d['focus_ratio_informative'])} (n={d['focus_ratio_informative']['n']}) |")
lines += ["", "| paired difference | WER | noise-focus ratio | top-patch share |", "|---|---|---|---|"]
for k, d in S["paired"].items():
    lines.append(f"| {k} | {ci(d['wer'], '{:+.3f}')} | {ci(d['focus_ratio'], '{:+.3f}')} | {ci(d['top_share'], '{:+.3f}')} |")
lines += ["", "| detector | " + " | ".join(MODELS) + " | paired base - moonshine |", "|---|" + "---|" * (len(MODELS) + 1)]
for det, d in S["detectors"].items():
    lines.append(f"| {det} | " + " | ".join(ci(d[m]) + f" (n={d[m]['n']})" for m in MODELS) + f" | {ci(d['paired whisper-base - moonshine-base'], '{:+.3f}')} |")
lines += ["", "| model | " + " | ".join(f"{edges[b] / 1000:.1f}-{edges[b + 1] / 1000:.1f} kHz" for b in range(8)) + " |", "|---|" + "---|" * 8]
for m in MODELS:
    lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in S["models"][m]["band_shares"]) + " |")
(RES / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
