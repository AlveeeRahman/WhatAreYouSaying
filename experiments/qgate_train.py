"""Can a small gate on audio patches lower the word error rate of a frozen speech model on noisy clips,
and does a quantum circuit do it better than a classical network of the same size?

Protocol, fixed before any test clip was scored:
  data      the 500-clip development slice, split 400 / 100 by seed 8001 for training and validation.
            The 100 test-slice clips of the other experiments are scored once per gate, at the end.
            The held-out slice is not read.
  gates     QuantumGate (4 qubits, 2 layers, CNOT ring), the same circuit without CNOTs, and a
            perceptron with about as many parameters (ways/qgate.py). All see the same four features.
  training  the speech model is frozen. Loss = mean surprisal of the human transcript under teacher
            forcing + 0.02 x mean(1 - g). Adam, learning rate 0.03, 4 epochs, batches of 8, seed 8001.
            The epoch with the lowest validation surprisal is kept.
  report    on the test clips: corpus word error rate and mean per-clip word error rate with a paired
            bootstrap against no gate, the mean surprisal of the human transcript ("confidence"), and
            the same on the clips where the detector flags at least half the frames.

Writes experiments/results/qgate/: summary.{json,md}, clips.csv, train_log.csv and one weights file per gate.
Usage: python experiments/qgate_train.py [data folder]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from common import AADD_CSV, AUDIO_DIR, DEV_CSV, REVISIONS, TRANSCRIPTS_TSV, data_root

from ways import erasure, load_backend
from ways.metrics import bootstrap_ci, paired_bootstrap, wer
from ways.qgate import ClassicalGate, QuantumGate, apply_gate, n_parameters, patch_features, reference_ids, reference_nll, save_gate, whisper_features
from ways.quality import flag_profile, load_scan_csv

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/qgate"
SEED, EPOCHS, BATCH, LR, KEEP_PENALTY, N_VAL = 8001, 4, 8, 0.03, 0.02, 100
MODELS = {"whisper-tiny": "openai/whisper-tiny", "whisper-base": "openai/whisper-base"}
GATES = {"quantum": lambda: QuantumGate(n_layers=2, entangle=True, seed=SEED),
         "quantum, no entanglement": lambda: QuantumGate(n_layers=2, entangle=False, seed=SEED),
         "classical": lambda: ClassicalGate(hidden=6, seed=SEED)}
NOISY_SHARE = 0.5


def load_clips(be, data, ids, gold, aadd):
    clips = []
    for cid in ids:
        wav, _ = sf.read(data / AUDIO_DIR / f"{cid}.wav", dtype="float32")
        x = torch.as_tensor(wav, device=be.device)
        grid = erasure.PatchGrid(len(wav), 0.1, 8)
        flags = flag_profile(aadd[cid], grid.n_t, 0.1)
        clips.append(dict(id=cid, wav=x, grid=grid, feat=patch_features(x, grid, flags), ref=reference_ids(be, gold[cid]), gold=gold[cid],
                          flagged=float(flags.mean())))
    return clips


def gated_features(be, gate, batch):
    """One circuit call for all patches of the batch, then each clip gated and turned into Whisper's input."""
    sizes = [c["feat"].shape[0] * c["feat"].shape[1] for c in batch]
    g_all = (torch.ones(sum(sizes), device=be.device) if gate is None
             else gate(torch.cat([c["feat"].reshape(-1, 4) for c in batch])))
    feats, gates, at = [], [], 0
    for c, n in zip(batch, sizes, strict=True):
        g = g_all[at: at + n].reshape(c["feat"].shape[:2]); at += n
        y = c["wav"] if gate is None else apply_gate(c["wav"], c["grid"], g)
        feats.append(whisper_features(be, y)); gates.append(g)
    return torch.stack(feats), gates


def nll_of(be, gate, clips):
    out = []
    with torch.no_grad():
        for i in range(0, len(clips), BATCH):
            feats, _ = gated_features(be, gate, clips[i: i + BATCH])
            out += reference_nll(be, feats, [c["ref"] for c in clips[i: i + BATCH]]).tolist()
    return np.array(out)


def train(be, gate, train_clips, val_clips, log, tag):
    opt = torch.optim.Adam(gate.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    best, best_state = np.inf, None
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_clips)); t0 = time.time(); losses, kept = [], []
        for i in range(0, len(order), BATCH):
            batch = [train_clips[j] for j in order[i: i + BATCH]]
            feats, gates = gated_features(be, gate, batch)
            keep = torch.cat([g.reshape(-1) for g in gates]).mean()
            nll = reference_nll(be, feats, [c["ref"] for c in batch]).mean()
            loss = nll + KEEP_PENALTY * (1 - keep)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(nll)); kept.append(float(keep))
        val = float(nll_of(be, gate, val_clips).mean())
        log.append(dict(run=tag, epoch=epoch + 1, train_nll=float(np.mean(losses)), val_nll=val, mean_gate=float(np.mean(kept)), seconds=time.time() - t0))
        print(tag, json.dumps(log[-1]), flush=True)
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in gate.state_dict().items()}
    gate.load_state_dict(best_state)
    return best


def score(be, gate, clips):
    rows = []
    nll = nll_of(be, gate, clips)
    with torch.no_grad():
        for c, s in zip(clips, nll, strict=True):
            if gate is None:
                y, kept = c["wav"], 1.0
            else:
                g = gate(c["feat"]); y = apply_gate(c["wav"], c["grid"], g); kept = float(g.mean())
            hyp = be.transcribe(y.cpu().numpy()).text
            w = wer(c["gold"], hyp)
            rows.append(dict(clip=c["id"], flagged=c["flagged"], wer=w["wer"], errors=w["S"] + w["D"] + w["I"], words=w["N"], nll=float(s), mean_gate=kept, text=hyp))
    return pd.DataFrame(rows)


def main():
    data = data_root(sys.argv[1] if len(sys.argv) > 1 else None, need=(AUDIO_DIR, DEV_CSV, TRANSCRIPTS_TSV, AADD_CSV))
    OUT.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(data / TRANSCRIPTS_TSV, sep="\t").set_index("id").normalized_text.astype(str).to_dict()
    aadd = load_scan_csv(data / AADD_CSV)
    dev = [str(f) for f in pd.read_csv(data / DEV_CSV).file]
    test = [str(f) for f in pd.read_csv(HERE / "sample100.csv").file]
    assert not set(dev) & set(test)
    perm = np.random.default_rng(SEED).permutation(len(dev))
    val_ids, train_ids = [dev[i] for i in perm[:N_VAL]], [dev[i] for i in perm[N_VAL:]]
    S = dict(protocol=dict(seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, keep_penalty=KEEP_PENALTY, n_train=len(train_ids), n_val=len(val_ids), n_test=len(test),
                           noisy_share=NOISY_SHARE), models={})
    log, all_rows = [], []
    for name, mid in MODELS.items():
        be = load_backend("hf-whisper", mid, revision=REVISIONS[mid])
        train_clips, val_clips, test_clips = (load_clips(be, data, ids, gold, aadd) for ids in (train_ids, val_ids, test))
        base = score(be, None, test_clips); base["gate"] = "none"; base["model"] = name; all_rows.append(base)
        noisy = (base.flagged >= NOISY_SHARE).to_numpy()
        M = dict(val_nll_no_gate=float(nll_of(be, None, val_clips).mean()), n_noisy_test_clips=int(noisy.sum()), gates={})

        def stats(d, base=base, noisy=noisy):
            out = {}
            for key, m in (("all", np.ones(len(d), bool)), ("noisy", noisy)):
                out[key] = dict(corpus_wer=float(d.errors[m].sum() / d.words[m].sum()), wer=bootstrap_ci(d.wer[m]), nll=bootstrap_ci(d.nll[m]),
                                wer_minus_no_gate=paired_bootstrap(d.wer[m], base.wer[m]), nll_minus_no_gate=paired_bootstrap(d.nll[m], base.nll[m]))
            return out

        M["gates"]["none"] = dict(parameters=0, **stats(base))
        for gname, make in GATES.items():
            torch.manual_seed(SEED)
            gate = make()
            best = train(be, gate, train_clips, val_clips, log, f"{name} / {gname}")
            save_gate(gate, OUT / f"{name}__{gname.replace(', ', '_').replace(' ', '_')}.json", model=mid, best_val_nll=best)
            d = score(be, gate, test_clips); d["gate"] = gname; d["model"] = name; all_rows.append(d)
            M["gates"][gname] = dict(parameters=n_parameters(gate), best_val_nll=best, mean_gate=float(d.mean_gate.mean()), **stats(d))
            print(name, gname, "test corpus WER", M["gates"][gname]["all"]["corpus_wer"], "no gate", M["gates"]["none"]["all"]["corpus_wer"], flush=True)
        S["models"][name] = M
        del be; torch.cuda.empty_cache()
    pd.concat(all_rows).to_csv(OUT / "clips.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "train_log.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    ci = lambda c, d=3: f"{c['mean']:+.{d}f} [{c['ci'][0]:+.{d}f}, {c['ci'][1]:+.{d}f}]"   # noqa: E731
    md = []
    for name, M in S["models"].items():
        for key, label in (("all", f"all {len(test)} test clips"), ("noisy", f"the {M['n_noisy_test_clips']} test clips with at least half their frames flagged")):
            md += [f"**{name}, {label}**", "", "| gate | parameters | corpus WER | per-clip WER minus no gate | surprisal of the human transcript (nats per token) | surprisal minus no gate | mean gate |",
                   "|---|---|---|---|---|---|---|"]
            for gname, G in M["gates"].items():
                r = G[key]
                md.append(f"| {gname} | {G['parameters']} | {r['corpus_wer']:.3f} | {'' if gname == 'none' else ci(r['wer_minus_no_gate'])} | {r['nll']['mean']:.3f} | "
                          f"{'' if gname == 'none' else ci(r['nll_minus_no_gate'])} | {G.get('mean_gate', 1.0):.2f} |")
            md.append("")
    (OUT / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
