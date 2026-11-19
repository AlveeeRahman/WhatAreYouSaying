"""Controlled degradation: clean LibriSpeech test-clean utterances with one known degradation inserted
into one known 2 s window, synthesised with AADD's own v3 synthesiser (aadd.v3.synth), then the same
clip scored clean and degraded by each model. Ground truth replaces the detector's opinion: we know
where the damage is, so we can ask directly whether the tokens spoken there still draw their
evidence from it, how much more surprised the model is by them, and what AADD flags.

Usage: python experiments/controlled.py prepare            # select clips, synthesise, run AADD v3
       python experiments/controlled.py run <name> <kind> <model id>   # one model, clean + degraded
       python experiments/controlled.py summarise
"""
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
LS = HERE / "cache/librispeech/LibriSpeech/test-clean"
CACHE = HERE / "cache/controlled"
OUT = HERE / "results/controlled"
SEED = 8001
N_CLIPS = 60
WIN_S = 2.0
CLASSES = ["noise", "reverb", "clipping", "lowpass", "hum", "clicks"]
SR = 16000
MODELS = {"whisper-tiny": ("hf-whisper", "openai/whisper-tiny"), "whisper-base": ("hf-whisper", "openai/whisper-base"),
          "moonshine-base": ("hf-moonshine", "UsefulSensors/moonshine-base")}


# ----------------------------------------------------------------------------- prepare
def _synth_rirs(rng, n=3):
    """Synthetic room impulse responses: direct path, a few early reflections, exponentially decaying
    diffuse tail with RT60 of 0.6 to 1.0 s (no recorded RIR corpus is needed)."""
    out = []
    for _ in range(n):
        rt60 = float(rng.uniform(0.6, 1.0))
        n_s = int(1.2 * SR)
        t = np.arange(n_s) / SR
        tail = rng.standard_normal(n_s).astype(np.float32) * np.exp(-6.9078 * t / rt60).astype(np.float32)
        h = np.zeros(n_s, np.float32)
        h[0] = 1.0
        for _ in range(6):
            h[int(rng.uniform(0.005, 0.05) * SR)] += float(rng.uniform(0.3, 0.7))
        h[int(0.01 * SR):] += 0.3 * tail[int(0.01 * SR):]
        out.append(h)
    return out


def prepare():
    from aadd.v3.synth import ORDER, Assets, Event, Synthesizer, _crossfade_mask, _rms, _active_rms

    assert LS.exists(), f"LibriSpeech test-clean not found at {LS}"
    rows = []
    for tr in sorted(LS.rglob("*.trans.txt")):
        for line in tr.read_text().splitlines():
            uid, text = line.split(" ", 1)
            f = tr.parent / f"{uid}.flac"
            info = sf.info(f)
            rows.append(dict(id=uid, path=str(f), text=text.lower(), duration=info.frames / info.samplerate))
    pool = pd.DataFrame(rows)
    pool = pool[(pool.duration >= 5) & (pool.duration <= 10)].sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(pool), N_CLIPS, replace=False))
    sel = pool.iloc[idx].reset_index(drop=True)
    sel["degradation"] = [CLASSES[i % len(CLASSES)] for i in range(len(sel))]
    sel["win_start"] = [float(rng.uniform(0.5, d - WIN_S - 0.5)) for d in sel.duration]
    sel["win_end"] = sel.win_start + WIN_S
    CACHE.mkdir(parents=True, exist_ok=True)
    assets = Assets(noises=[], music=[], rirs=_synth_rirs(rng), babble=None)

    class Controlled(Synthesizer):
        def _noise(self, x, ev, rng_, srms):           # pink noise at a fixed 0 dB SNR, no corpus needed
            n = rng_.standard_normal(self.n).astype(np.float32)
            f = np.fft.rfftfreq(self.n, 1 / SR); f[0] = f[1]
            n = np.fft.irfft(np.fft.rfft(n) / np.sqrt(f), n=self.n).astype(np.float32)
            ev.params.update(source="colour:pink", snr_db=0.0)
            return x + srms / (_rms(n) + 1e-9) * n, None

        def make_pair(self, speech, rng_, events):
            """Like make(), but with a fixed level (peak 0.6) so the clean twin is the exact same signal."""
            x = (speech * (0.6 / (np.abs(speech).max() + 1e-9))).astype(np.float32)
            clean = x.copy()
            F, C = self.n_frames, len(self.class_p)
            target = np.zeros((F, C), np.float32); ignore = np.zeros((F, C), bool)
            srms = _active_rms(clean)
            fe = (clean[: F * 320].reshape(F, 320) ** 2).mean(1)
            silent = fe < (srms ** 2) * 10 ** (-40 / 10)
            for cls in ORDER:
                for ev in [e for e in events if e.cls == cls]:
                    y, obs = getattr(self, f"_{cls}")(x, ev, rng_, srms)
                    m = _crossfade_mask(self.n, ev.start, ev.end)
                    x = (x * (1 - m) + y * m).astype(np.float32)
                    self._label(target, ignore, ev, obs, silent)
            return clean, np.clip(x, -1, 1).astype(np.float32), target, ignore

    params = {}
    for r in sel.itertuples():
        wav, sr = sf.read(r.path, dtype="float32")
        assert sr == SR
        synth = Controlled(assets, seconds=len(wav) / SR)
        ev = Event(r.degradation, int(r.win_start * SR), int(r.win_end * SR))
        clean, deg, target, ignore = synth.make_pair(wav, np.random.default_rng(SEED + r.Index), [ev])
        sf.write(CACHE / f"{r.id}_clean.wav", clean, SR)
        sf.write(CACHE / f"{r.id}_deg.wav", deg, SR)
        np.save(CACHE / f"{r.id}_target.npy", target)
        params[r.id] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in ev.params.items()}
    sel["params"] = sel.id.map(lambda i: json.dumps(params[i]))
    sel.to_csv(CACHE / "manifest.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    sel.drop(columns=["path"]).to_csv(OUT / "manifest.csv", index=False)
    # AADD v3 on the degraded clips, against the ground-truth window
    from ways.quality import CLASSES as AADD_CLASSES, Quality
    q = Quality("v3")
    det = []
    for r in sel.itertuples():
        deg, _ = sf.read(CACHE / f"{r.id}_deg.wav", dtype="float32")
        P = q.frames(deg)
        np.save(CACHE / f"{r.id}_aadd.npy", P)
        n = len(P); t = np.arange(n) * 0.02
        inside = (t >= r.win_start) & (t < r.win_end)
        j = AADD_CLASSES.index(r.degradation)
        det.append(dict(id=r.id, degradation=r.degradation, flagged_inside=float((P[inside] >= 0.5).any(1).mean()),
                        flagged_outside=float((P[~inside] >= 0.5).any(1).mean()),
                        class_inside=float((P[inside, j] >= 0.5).mean()), class_outside=float((P[~inside, j] >= 0.5).mean()),
                        top_class_inside=AADD_CLASSES[int(P[inside].mean(0).argmax())]))
    pd.DataFrame(det).to_csv(OUT / "aadd_detection.csv", index=False)
    print(pd.DataFrame(det).groupby("degradation")[["flagged_inside", "flagged_outside", "class_inside", "class_outside"]].mean().round(2))


# ----------------------------------------------------------------------------- run
def run(name, kind, model_id):
    from ways import explain, load_backend
    from ways.metrics import wer
    from common import REVISIONS

    man = pd.read_csv(CACHE / "manifest.csv")
    d = OUT / name; (d / "npz").mkdir(parents=True, exist_ok=True)
    rows_f = d / "rows.jsonl"
    done = {(json.loads(l)["id"], json.loads(l)["cond"]) for l in open(rows_f)} if rows_f.exists() else set()
    be = load_backend(kind, model_id, revision=REVISIONS.get(model_id))
    for r in man.itertuples():
        for cond in ("clean", "deg"):
            if (r.id, cond) in done:
                continue
            t0 = time.time()
            wav, _ = sf.read(CACHE / f"{r.id}_{cond}.wav", dtype="float32")
            att = explain(be, wav, batch=64)
            att.save(d / "npz" / f"{r.id}_{cond}.npz")
            w = wer(r.text, att.text)
            row = dict(id=r.id, cond=cond, degradation=r.degradation, model=name, text=att.text, wer=w["wer"], n_tokens=int(att.delta.shape[0]),
                       mean_surprisal0=float(att.surprisal0.mean()), seconds=time.time() - t0)
            with open(rows_f, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{name} {r.id} {cond} WER {w['wer']:.2f} {row['seconds']:.1f}s", flush=True)
    from ways.provenance import write as write_manifest
    write_manifest(d / "provenance.json", inputs={"manifest.csv": str(OUT / "manifest.csv"), "aadd_detection.csv": str(OUT / "aadd_detection.csv")},
                   seeds={"sample": SEED, "synth": SEED, "bootstrap": 8001}, models={model_id: REVISIONS.get(model_id)},
                   extra=dict(backend=be.describe(), patch_s=0.1, n_bands=8, n_clips=int(len(man)), conditions=["clean", "deg"]))


# ----------------------------------------------------------------------------- summarise
def _mass_in_window(att, a, b):
    from ways.metrics import token_mass
    m = token_mass(att); prof = np.nansum(m, axis=1)          # [K, T]
    t0 = np.arange(att.n_t) * att.patch_s; t1 = t0 + att.patch_s
    f = np.clip((np.minimum(t1, b) - np.maximum(t0, a)) / att.patch_s, 0, 1)   # overlap fraction per patch
    return prof @ f, f.mean()


def summarise():
    from ways.attribution import Attribution
    from ways.metrics import bootstrap_ci, paired_bootstrap

    man = pd.read_csv(OUT / "manifest.csv").set_index("id")
    S = dict(design=dict(n_clips=int(len(man)), window_s=WIN_S, classes=CLASSES, seed=SEED, source="LibriSpeech test-clean, 5-10 s utterances",
                         synth="aadd.v3.synth.Synthesizer (noise: pink at 0 dB SNR; reverb: synthetic RIR, RT60 0.6-1.0 s; others: AADD defaults)"),
             aadd=pd.read_csv(OUT / "aadd_detection.csv").groupby("degradation")[["flagged_inside", "flagged_outside", "class_inside", "class_outside"]].mean().round(3).to_dict("index"),
             models={})
    tok_rows, clip_rows = [], []
    for name in MODELS:
        d = OUT / name
        if not (d / "rows.jsonl").exists():
            continue
        rows = pd.DataFrame([json.loads(l) for l in open(d / "rows.jsonl")]).drop_duplicates(["id", "cond"], keep="last")
        for cid, g in rows.groupby("id"):
            if set(g.cond) != {"clean", "deg"}:
                continue
            a, b = man.loc[cid, "win_start"], man.loc[cid, "win_end"]
            ac, ad = Attribution.load(d / "npz" / f"{cid}_clean.npz"), Attribution.load(d / "npz" / f"{cid}_deg.npz")
            mc, fbar = _mass_in_window(ac, a, b); md, _ = _mass_in_window(ad, a, b)
            sm = SequenceMatcher(None, list(ac.token_ids), list(ad.token_ids), autojunk=False)
            pairs = [(i + o, j + o) for i, j, n in sm.get_matching_blocks() for o in range(n)]
            in_win = {i for i in range(len(mc)) if np.isfinite(mc[i]) and mc[i] >= 0.5}   # token leaned on the window when clean
            wc = float(g[g.cond == "clean"].wer.iloc[0]); wd = float(g[g.cond == "deg"].wer.iloc[0])
            cr = dict(model=name, id=cid, degradation=man.loc[cid, "degradation"], wer_clean=wc, wer_deg=wd, n_clean=len(mc), n_deg=len(md),
                      n_matched=len(pairs), n_in_window=len(in_win), window_share=float(fbar))
            for i, j in pairs:
                tok_rows.append(dict(model=name, id=cid, degradation=cr["degradation"], k_clean=i, k_deg=j, piece=ac.pieces[i],
                                     in_window=i in in_win, mass_clean=float(mc[i]), mass_deg=float(md[j]),
                                     s_clean=float(ac.surprisal0[i]), s_deg=float(ad.surprisal0[j])))
            sub = [t for t in tok_rows if t["model"] == name and t["id"] == cid]
            for grp, flag in (("in", True), ("out", False)):
                ss = [t for t in sub if t["in_window"] == flag and np.isfinite(t["mass_clean"]) and np.isfinite(t["mass_deg"])]
                if ss:
                    cr[f"mass_clean_{grp}"] = float(np.mean([t["mass_clean"] for t in ss])); cr[f"mass_deg_{grp}"] = float(np.mean([t["mass_deg"] for t in ss]))
                    cr[f"ds_{grp}"] = float(np.mean([t["s_deg"] - t["s_clean"] for t in ss]))
            cr["kept_frac"] = len(pairs) / max(len(mc), 1)
            clip_rows.append(cr)
    clips = pd.DataFrame(clip_rows); toks = pd.DataFrame(tok_rows)
    clips.to_csv(OUT / "clips.csv", index=False); toks.to_csv(OUT / "tokens.csv", index=False)
    for name in MODELS:
        c = clips[clips.model == name]
        if not len(c):
            continue
        M = dict(n_clips=int(len(c)), wer_clean=bootstrap_ci(c.wer_clean), wer_deg=bootstrap_ci(c.wer_deg), wer_paired=paired_bootstrap(c.wer_deg, c.wer_clean),
                 kept_tokens=bootstrap_ci(c.kept_frac),
                 in_window=dict(mass_clean=bootstrap_ci(c.mass_clean_in), mass_deg=bootstrap_ci(c.mass_deg_in), mass_paired=paired_bootstrap(c.mass_deg_in, c.mass_clean_in), surprisal_rise=bootstrap_ci(c.ds_in)),
                 out_window=dict(mass_clean=bootstrap_ci(c.mass_clean_out), mass_deg=bootstrap_ci(c.mass_deg_out), mass_paired=paired_bootstrap(c.mass_deg_out, c.mass_clean_out), surprisal_rise=bootstrap_ci(c.ds_out)),
                 by_class={})
        for cls, g in c.groupby("degradation"):
            M["by_class"][cls] = dict(n=int(len(g)), wer_clean=float(g.wer_clean.mean()), wer_deg=float(g.wer_deg.mean()),
                                      mass_in_clean=float(g.mass_clean_in.mean()), mass_in_deg=float(g.mass_deg_in.mean()), ds_in=float(g.ds_in.mean()), ds_out=float(g.ds_out.mean()))
        S["models"][name] = M
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    ci = lambda d, f="{:.2f}": f"{f.format(d['mean'])} [{f.format(d['ci'][0])}, {f.format(d['ci'][1])}]"
    L = ["| model | WER clean | WER degraded | paired rise | tokens kept | in-window tokens: mass in window, clean -> degraded | paired change | surprisal rise in / out |", "|---|---|---|---|---|---|---|---|"]
    for n, M in S["models"].items():
        L.append(f"| {n} | {ci(M['wer_clean'])} | {ci(M['wer_deg'])} | {ci(M['wer_paired'], '{:+.3f}')} | {ci(M['kept_tokens'])} | {ci(M['in_window']['mass_clean'])} -> {ci(M['in_window']['mass_deg'])} | {ci(M['in_window']['mass_paired'], '{:+.3f}')} | {ci(M['in_window']['surprisal_rise'])} / {ci(M['out_window']['surprisal_rise'])} |")
    L += ["", "| degradation | AADD flags inside | outside | right class inside | " + " | ".join(f"{n}: WER clean -> deg, mass-in clean -> deg" for n in S["models"]) + " |", "|---|---|---|---|" + "---|" * len(S["models"])]
    for cls in CLASSES:
        a = S["aadd"].get(cls, {})
        L.append(f"| {cls} | {a.get('flagged_inside', float('nan')):.2f} | {a.get('flagged_outside', float('nan')):.2f} | {a.get('class_inside', float('nan')):.2f} | " +
                 " | ".join(f"{M['by_class'][cls]['wer_clean']:.2f} -> {M['by_class'][cls]['wer_deg']:.2f}, {M['by_class'][cls]['mass_in_clean']:.2f} -> {M['by_class'][cls]['mass_in_deg']:.2f}" if cls in M["by_class"] else "" for M in S["models"].values()) + " |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "run":
        run(*sys.argv[2:5])
    elif cmd == "summarise":
        summarise()
