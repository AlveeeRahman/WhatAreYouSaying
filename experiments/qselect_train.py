"""Can a small scorer pick, clip by clip, the best of three speech models' transcripts, and does a
quantum circuit do it better than a classical network of the same size?

The patch gate (experiments/qgate_train.py) changes the audio and did not help. This experiment
leaves the audio alone and uses what the models already disagree on. Every clip goes through
whisper-tiny, whisper-base and moonshine-base. Each transcript is described by four numbers:

  surprisal      the model's mean surprisal of its own transcript (nats per token)
  disagreement   the transcript's mean word error rate against the other two models' transcripts
  rate           its length in words relative to the median of the three (a looping model writes too much)
  flagged        the share of the clip the degradation detector flags

A scorer maps those four numbers, plus one learned offset per model, to a predicted 1 - WER, and the
transcript with the highest score is kept. Three scorers share that set-up: the four-qubit circuit
of ways/qgate.py, the same circuit without entanglement, and a perceptron of similar size.

Protocol, fixed before any test clip was scored: the scorers are fitted on 400 development clips
(mean squared error, Adam, learning rate 0.05, 300 full-batch steps, seed 8001), the step with the
lowest error on the other 100 development clips is kept, and the 100 test-slice clips are scored
once. Reference points on the test clips: each model alone, the transcript with the lowest
surprisal, the transcript closest to the other two, and the oracle that always picks the best.

Writes experiments/results/qselect/: summary.{json,md}, and hypotheses.csv, a cache of the transcripts that is not committed. The fitted scorers go to ways/gates/, where the q-select tool finds them.
Usage: python experiments/qselect_train.py [data folder]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from common import AADD_CSV, AUDIO_DIR, DEV_CSV, REVISIONS, TRANSCRIPTS_TSV, data_root

from ways import load_backend
from ways.metrics import paired_bootstrap, wer
from ways.qgate import ClassicalGate, QuantumGate, n_parameters
from ways.qselect import GATES, Scorer, hypothesis_features, save_scorer
from ways.quality import flagged_mask, load_scan_csv

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/qselect"
SEED, STEPS, LR, N_VAL, NOISY_SHARE = 8001, 300, 0.05, 100, 0.5
MODELS = {"whisper-tiny": ("hf-whisper", "openai/whisper-tiny"), "whisper-base": ("hf-whisper", "openai/whisper-base"),
          "moonshine-base": ("hf-moonshine", "UsefulSensors/moonshine-base")}
NAMES = list(MODELS)
SCORERS = {"quantum": lambda: QuantumGate(n_layers=2, entangle=True, seed=SEED),
           "quantum, no entanglement": lambda: QuantumGate(n_layers=2, entangle=False, seed=SEED),
           "classical": lambda: ClassicalGate(hidden=6, seed=SEED)}


def hypotheses(data, clips, gold, aadd):
    f = OUT / "hypotheses.csv"
    if f.exists():
        return pd.read_csv(f, keep_default_na=False)
    rows = []
    for name, (kind, mid) in MODELS.items():
        be = load_backend(kind, mid, revision=REVISIONS[mid])
        for split, ids in clips.items():
            for cid in ids:
                wav, _ = sf.read(data / AUDIO_DIR / f"{cid}.wav", dtype="float32")
                tr = be.transcribe(wav)
                s = be.surprisal(wav[None], tr)[0][0] if tr.n_tokens else np.zeros(1)
                w = wer(gold[cid], tr.text)
                rows.append(dict(split=split, clip=cid, model=name, text=tr.text, surprisal=float(s.mean()), wer=w["wer"], errors=w["S"] + w["D"] + w["I"],
                                 words=w["N"], flagged=float(flagged_mask(aadd[cid]).mean())))
        print("transcribed", name, flush=True)
        del be; torch.cuda.empty_cache()
    d = pd.DataFrame(rows); d.to_csv(f, index=False)
    return d


def features(d):
    """One row per (clip, model): the four features of ways.qselect, the target 1 - WER, and the error counts."""
    out = []
    for cid, g in d.groupby("clip", sort=False):
        g = g.set_index("model").loc[NAMES]
        x = hypothesis_features(list(g.text), list(g.surprisal), float(g.flagged.iloc[0]))
        for i, m in enumerate(NAMES):
            out.append(dict(clip=cid, model=m, m=i, f_surprisal=x[i, 0], f_disagree=x[i, 1], f_rate=x[i, 2], f_flagged=x[i, 3],
                            target=1 - min(g.wer[m], 1.0), wer=g.wer[m], errors=g.errors[m], words=g.words[m], flagged=g.flagged[m]))
    return pd.DataFrame(out)


def tensors(f):
    x = torch.tensor(f[["f_surprisal", "f_disagree", "f_rate", "f_flagged"]].to_numpy(np.float32))
    return x, torch.tensor(f.m.to_numpy()), torch.tensor(f.target.to_numpy(np.float32))


def fit(scorer, tr, va):
    opt = torch.optim.Adam(scorer.parameters(), lr=LR)
    (xt, mt, yt), (xv, mv, yv) = tensors(tr), tensors(va)
    best, state = np.inf, None
    for _ in range(STEPS):
        loss = ((scorer(xt, mt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((scorer(xv, mv) - yv) ** 2).mean())
        if v < best:
            best, state = v, {k: p.detach().clone() for k, p in scorer.state_dict().items()}
    scorer.load_state_dict(state)
    return best


def picked(f, score):
    """The rows of the transcript with the highest score in every clip, in clip order."""
    f = f.assign(score=np.asarray(score))
    return f.loc[f.groupby("clip", sort=False).score.idxmax()].reset_index(drop=True)


def main():
    data = data_root(sys.argv[1] if len(sys.argv) > 1 else None, need=(AUDIO_DIR, DEV_CSV, TRANSCRIPTS_TSV, AADD_CSV))
    OUT.mkdir(parents=True, exist_ok=True); GATES.mkdir(exist_ok=True)
    gold = pd.read_csv(data / TRANSCRIPTS_TSV, sep="\t").set_index("id").normalized_text.astype(str).to_dict()
    aadd = load_scan_csv(data / AADD_CSV)
    dev = [str(f) for f in pd.read_csv(data / DEV_CSV).file]
    test = [str(f) for f in pd.read_csv(HERE / "sample100.csv").file]
    assert not set(dev) & set(test)
    perm = np.random.default_rng(SEED).permutation(len(dev))
    clips = dict(val=[dev[i] for i in perm[:N_VAL]], train=[dev[i] for i in perm[N_VAL:]], test=test)
    hyp = hypotheses(data, clips, gold, aadd)
    F = {s: features(hyp[hyp.split == s]) for s in clips}
    te = F["test"]
    reference = te[te.model == "whisper-base"].reset_index(drop=True)
    noisy = (reference.flagged >= NOISY_SHARE).to_numpy()
    rows = {}

    def record(label, sel, parameters=0, **extra):
        r = dict(parameters=parameters, **extra)
        for key, m in (("all", np.ones(len(sel), bool)), ("noisy", noisy)):
            r[key] = dict(corpus_wer=float(sel.errors[m].sum() / sel.words[m].sum()), mean_wer=float(sel.wer[m].mean()),
                          wer_minus_whisper_base=paired_bootstrap(sel.wer[m], reference.wer[m]))
        r["share_picked"] = {n: float((sel.model == n).mean()) for n in NAMES}
        rows[label] = r

    for n in NAMES:
        record(f"{n} alone", te[te.model == n].reset_index(drop=True))
    record("lowest surprisal", picked(te, -te.f_surprisal))
    record("closest to the other two", picked(te, -te.f_disagree))
    record("oracle", picked(te, -te.wer))
    for sname, make in SCORERS.items():
        torch.manual_seed(SEED)
        scorer = Scorer(make(), NAMES)
        best = fit(scorer, F["train"], F["val"])
        with torch.no_grad():
            x, m, _ = tensors(te)
            sel = picked(te, scorer(x, m).numpy())
        save_scorer(scorer, GATES / f"qselect__{sname.replace(', ', '_').replace(' ', '_')}.json", fitted_on="400 development clips", seed=SEED, val_mse=best)
        record(f"scorer: {sname}", sel, parameters=n_parameters(scorer), val_mse=best)
        print(sname, json.dumps(rows[f"scorer: {sname}"]["all"]), flush=True)
    S = dict(protocol=dict(seed=SEED, steps=STEPS, lr=LR, n_train=len(clips["train"]), n_val=N_VAL, n_test=len(test), noisy_share=NOISY_SHARE,
                           n_noisy_test_clips=int(noisy.sum())), rows=rows)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    ci = lambda c: f"{c['mean']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"   # noqa: E731
    md = [f"| transcript kept | parameters | corpus WER, {len(test)} test clips | per-clip WER minus whisper-base | corpus WER, {int(noisy.sum())} noisy clips | per-clip WER minus whisper-base, noisy | picked tiny / base / moonshine |",
          "|---|---|---|---|---|---|---|"]
    for label, r in rows.items():
        p = r["share_picked"]
        md.append(f"| {label} | {r['parameters'] or ''} | {r['all']['corpus_wer']:.3f} | {ci(r['all']['wer_minus_whisper_base'])} | {r['noisy']['corpus_wer']:.3f} | "
                  f"{ci(r['noisy']['wer_minus_whisper_base'])} | {p[NAMES[0]]:.0%} / {p[NAMES[1]]:.0%} / {p[NAMES[2]]:.0%} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
