"""Command line.

  ways explain clip.wav --backend hf-whisper --model openai/whisper-base --out out/
  ways explain clip.wav --backend hf-moonshine --model UsefulSensors/moonshine-base --out out/
  ways explain clip.wav --backend openai-compat --base-url http://localhost:8000/v1 --model openai/whisper-large-v3 --out out/
  ways quality clip.wav                      # AADD v3 degradation timeline, 20 ms
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load(path):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def cmd_explain(a):
    from . import explain, load_backend
    from .metrics import clip_summary, token_summary

    wav = _load(a.audio)
    kw = {}
    if a.backend == "openai-compat":
        kw = dict(base_url=a.base_url, route=a.route)
    be = load_backend(a.backend, a.model, **kw)
    att = explain(be, wav, patch_s=a.patch, n_bands=a.bands, batch=a.batch, log=print if a.verbose else None)
    probs = None
    if a.aadd:
        from .quality import Quality

        probs = Quality(a.aadd_model).frames(wav)
    rows = token_summary(att, probs)
    summ = clip_summary(att, rows, probs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    att.save(out / f"{stem}.npz")
    (out / f"{stem}.json").write_text(json.dumps(dict(summary=summ, tokens=rows), indent=1, default=float))
    if probs is not None:
        np.save(out / f"{stem}_aadd.npy", probs)
    if a.figure:
        from .plots import clip_figure

        clip_figure(att, probs, wav, out / f"{stem}.png", title=f"{stem}  ·  {att.model}")
    print(json.dumps(summ, indent=1, default=float))
    print("transcript:", att.text)
    for r in rows:
        print(f"  {r['k']:3d} {r['piece']!r:16} top {r.get('top_centre_s', float('nan')):5.2f}s band {r.get('top_band', -1)} "
              f"share {r.get('top_share', float('nan')):.2f} focus {r.get('focus_ratio', float('nan')):.2f}")


def cmd_quality(a):
    from .quality import CLASSES, Quality, intervals

    wav = _load(a.audio)
    probs = Quality(a.aadd_model).frames(wav)
    fl = (probs >= 0.5).any(1)
    print(f"{len(probs)} frames of 20 ms, {fl.mean():.0%} flagged at p >= 0.5")
    for name, s, e, pk in intervals(probs):
        print(f"  {name:13} {s:6.2f}-{e:6.2f} s  peak {pk:.2f}")
    if a.out:
        np.save(a.out, probs)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ways", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explain", help="per-token erasure attribution for one clip")
    e.add_argument("audio")
    e.add_argument("--backend", default="hf-whisper", choices=["hf-whisper", "hf-moonshine", "openai-compat"])
    e.add_argument("--model", default=None)
    e.add_argument("--base-url", default="http://localhost:8000/v1")
    e.add_argument("--route", default="transcriptions", choices=["transcriptions", "chat"])
    e.add_argument("--patch", type=float, default=0.1, help="patch length in seconds (multiple of 0.01)")
    e.add_argument("--bands", type=int, default=8)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--aadd", action="store_true", help="also run AADD and report the noise-focus ratio")
    e.add_argument("--aadd-model", default="v3")
    e.add_argument("--figure", action="store_true")
    e.add_argument("--out", default="ways_out")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(fn=cmd_explain)
    q = sub.add_parser("quality", help="AADD degradation timeline for one clip")
    q.add_argument("audio")
    q.add_argument("--aadd-model", default="v3")
    q.add_argument("--out", default=None)
    q.set_defaults(fn=cmd_quality)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
