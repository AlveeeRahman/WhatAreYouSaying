"""Does scoring with the fused attention kernel give the same maps as the step-by-step one?

The saved maps of the 100 test-slice clips were made with the "eager" attention. This script scores
the same clips again with the fused kernel (the default from 0.3.3) and compares, token by token:
the unerased surprisal, every delta, every flip, the top patch, and per clip the noise-focus ratio.
It also reports the time per clip of both. Nothing saved is overwritten.

Needs the saved maps (experiments/run_all.sh). Writes experiments/results/checks/fast_attention.{json,md}.
Usage: python experiments/check_fast_attention.py [data folder]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from common import AADD_CSV, AUDIO_DIR, REVISIONS, data_root

from ways import explain, load_backend
from ways.attribution import Attribution
from ways.metrics import clip_summary, token_summary
from ways.quality import load_scan_csv

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "checks"
MODELS = {"whisper-tiny": "openai/whisper-tiny", "whisper-base": "openai/whisper-base"}


def focus(att, probs):
    return clip_summary(att, token_summary(att, probs), probs).get("focus_ratio_mean", float("nan"))


def main():
    data = data_root(sys.argv[1] if len(sys.argv) > 1 else None, need=(AUDIO_DIR, AADD_CSV))
    OUT.mkdir(parents=True, exist_ok=True)
    aadd = load_scan_csv(data / AADD_CSV)
    clips = list(pd.read_csv(HERE / "sample100.csv").file)
    S = dict(n_clips=len(clips), models={})
    for name, mid in MODELS.items():
        be = load_backend("hf-whisper", mid, revision=REVISIONS[mid])
        assert be.fast_attention
        rows = []
        for i, clip in enumerate(clips):
            old = Attribution.load(RES / name / "npz" / f"{clip}.npz")
            wav, _ = sf.read(data / AUDIO_DIR / f"{clip}.wav", dtype="float32")
            t0 = time.time(); new = explain(be, wav, batch=64); seconds = time.time() - t0
            if new.token_ids != old.token_ids:
                rows.append(dict(clip=clip, same_transcript=False)); continue
            scored = old.delta.reshape(len(old.token_ids), -1).max(1) > 0
            top_old = old.delta.reshape(len(scored), -1).argmax(1); top_new = new.delta.reshape(len(scored), -1).argmax(1)
            f_old, f_new = focus(old, aadd[clip]), focus(new, aadd[clip])
            rows.append(dict(clip=clip, same_transcript=True, tokens=int(scored.sum()), patches=int(old.delta[0].size),
                             max_abs_delta=float(np.abs(new.delta - old.delta).max()), max_abs_s0=float(np.abs(new.surprisal0 - old.surprisal0).max()),
                             flips_differ=int((new.flip != old.flip).sum()), flips_total=int(old.flip.size),
                             top_same=int((top_old == top_new)[scored].sum()), focus_old=f_old, focus_new=f_new,
                             seconds_eager=float(old.seconds), seconds_fused=float(seconds)))
            if (i + 1) % 20 == 0:
                print(name, i + 1, "clips", flush=True)
        d = pd.DataFrame(rows); d.to_csv(OUT / f"fast_attention__{name}.csv", index=False)
        ok = d[d.same_transcript]
        dfocus = (ok.focus_new - ok.focus_old).abs()
        S["models"][name] = dict(clips_same_transcript=int(len(ok)), tokens=int(ok.tokens.sum()), patch_scores=int((ok.tokens * ok.patches).sum()),
                                 top_patch_same=int(ok.top_same.sum()), max_abs_delta_nats=float(ok.max_abs_delta.max()),
                                 median_clip_max_abs_delta_nats=float(ok.max_abs_delta.median()), max_abs_surprisal0_nats=float(ok.max_abs_s0.max()),
                                 flips_differ=int(ok.flips_differ.sum()), flips_total=int(ok.flips_total.sum()),
                                 max_abs_focus_ratio_change=float(dfocus.max()), seconds_per_clip_eager=float(ok.seconds_eager.mean()),
                                 seconds_per_clip_fused=float(ok.seconds_fused.mean()), speedup=float(ok.seconds_eager.sum() / ok.seconds_fused.sum()))
        print(name, json.dumps(S["models"][name]), flush=True)
    (OUT / "fast_attention.json").write_text(json.dumps(S, indent=1))
    md = ["| model | clips with the same transcript | tokens | top patch unchanged | largest change of any delta (nats) | flips that differ | largest change of a clip's noise-focus ratio | s per clip, eager | s per clip, fused | speed-up |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in S["models"].items():
        md.append(f"| {name} | {m['clips_same_transcript']} of {S['n_clips']} | {m['tokens']} | {m['top_patch_same']} of {m['tokens']} | {m['max_abs_delta_nats']:.1e} | "
                  f"{m['flips_differ']} of {m['flips_total']} | {m['max_abs_focus_ratio_change']:.1e} | {m['seconds_per_clip_eager']:.1f} | {m['seconds_per_clip_fused']:.1f} | {m['speedup']:.2f}x |")
    (OUT / "fast_attention.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
