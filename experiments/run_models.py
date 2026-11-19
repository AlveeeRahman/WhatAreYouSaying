"""Run one backend over the 100-clip sample: transcribe, WER against the human transcript,
erasure attribution over every 100 ms x band patch, per-token and per-clip summaries with the AADD
flags as context. Resumable; one line per clip in results/<name>/rows.jsonl.

Usage: python experiments/run_models.py <name> <backend kind> <model id> [data folder]
  e.g. python experiments/run_models.py whisper-base hf-whisper openai/whisper-base
       python experiments/run_models.py moonshine-base hf-moonshine UsefulSensors/moonshine-base
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from common import AADD_CSV, AUDIO_DIR, REVISIONS, TRANSCRIPTS_TSV, data_root

from ways import explain, load_backend
from ways.metrics import clip_summary, token_summary, wer
from ways.provenance import write as write_manifest
from ways.quality import load_scan_csv

name, kind, model_id = sys.argv[1:4]
HERE = Path(__file__).resolve().parent
DATA = data_root(sys.argv[4] if len(sys.argv) > 4 else None, need=(AUDIO_DIR, TRANSCRIPTS_TSV, AADD_CSV))
AUDIO = DATA / AUDIO_DIR
GOLD = DATA / TRANSCRIPTS_TSV
AADD_CSV = DATA / AADD_CSV
OUT = HERE / "results" / name
(OUT / "npz").mkdir(parents=True, exist_ok=True)
log = open(OUT / "run.log", "a")


def say(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n"); log.flush()


sample = pd.read_csv(HERE / "sample100.csv")
gold = pd.read_csv(GOLD, sep="\t", usecols=["id", "normalized_text"], keep_default_na=False)
gold = dict(zip(gold.id, gold.normalized_text, strict=True))
say("loading AADD frame probabilities from", AADD_CSV.name)
aadd = load_scan_csv(AADD_CSV)
be = load_backend(kind, model_id, revision=REVISIONS.get(model_id))
say("backend", json.dumps(be.describe()))
rows_f = OUT / "rows.jsonl"
done = {json.loads(l)["clip"] for l in open(rows_f)} if rows_f.exists() else set()
tok_rows = []
for i, r in sample.iterrows():
    clip = r["file"]
    if clip in done:
        continue
    t0 = time.time()
    wav, sr = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == 16000
    att = explain(be, wav, patch_s=0.1, n_bands=8, batch=64)
    att.save(OUT / "npz" / f"{clip}.npz")
    P = aadd.get(clip)
    rows = token_summary(att, P)
    cs = clip_summary(att, rows, P)
    w = wer(gold.get(clip, ""), att.text)
    cs.update(clip=clip, label=r["label"], original_index=int(r["original_index"]), text=att.text, gold=gold.get(clip, ""),
              **{f"wer_{k}": v for k, v in w.items()}, wer=w["wer"])
    with open(rows_f, "a") as f:
        f.write(json.dumps(cs, default=float) + "\n")
    for tr in rows:
        tr.update(clip=clip, model=name)
    pd.DataFrame(rows).to_csv(OUT / "npz" / f"{clip}_tokens.csv", index=False)
    say(f"[{i + 1}/{len(sample)}] {clip} K={att.delta.shape[0]} patches={att.n_bands * att.n_t} WER={w['wer']:.2f} "
        f"focus={cs.get('focus_ratio_mean', float('nan')):.2f} {time.time() - t0:.1f}s")

# tables
df = pd.DataFrame([json.loads(l) for l in open(rows_f)]).drop_duplicates("clip", keep="last")
df.to_csv(OUT / "clips.csv", index=False)
toks = pd.concat([pd.read_csv(p) for p in sorted((OUT / "npz").glob("*_tokens.csv"))], ignore_index=True)
toks.to_csv(OUT / "tokens.csv", index=False)
write_manifest(OUT / "provenance.json", inputs={"sample100.csv": str(HERE / "sample100.csv"), "transcripts_tsv": str(GOLD),
                                                 "aadd_frames_csv": str(AADD_CSV)},
               seeds={"sample": 8001, "bootstrap": 8001}, models={model_id: REVISIONS.get(model_id)},
               extra=dict(backend=be.describe(), patch_s=0.1, n_bands=8, n_clips=int(len(df))))
say("done", name, "clips", len(df), "tokens", len(toks))
