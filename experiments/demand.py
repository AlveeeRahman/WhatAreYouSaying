"""Real recorded noise on an open licence: the VoiceBank-DEMAND test set (Valentini-Botinhao et al.,
Edinburgh DataShare 10283/2791, CC BY 4.0). 824 VCTK sentences from two speakers, each released clean
and with one DEMAND noise recording (bus, cafe, living room, office, public square) mixed in at 2.5,
7.5, 12.5 or 17.5 dB SNR, sample-aligned, with the sentence text.

Three conditions per clip, 100 clips drawn at random (default_rng(SEED)):
  clean    the studio recording
  noisy    the released noisy twin (noise over the whole clip)
  alt      the noise switched on and off: the clip is cut into 4 to 7 segments of about 0.4 to 0.9 s and
           each segment is taken from the clean or the noisy twin following a random pattern with at
           least two noisy runs and two clean runs (10 ms crossfades), so "noisy, clean, noisy, clean,
           clean, noisy" within one sentence, with real noise and the exact boundaries known.

Per model: WER per condition, and on the alternating clips whether each word's evidence sits in the
segment where the word is spoken. The truth for where a word is spoken is whisper-base's word
timestamps on the clean recording, so one truth serves every model.

Usage: python experiments/demand.py prepare
       python experiments/demand.py run <name> <kind> <model id> [max_minutes]
       python experiments/demand.py summarise
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
CACHE = HERE / "cache/demand"
OUT = HERE / "results/demand"
SEED = 8001
N_CLIPS = 100
SR = 16000
SEG_RANGE = (0.4, 0.9)
XFADE_S = 0.01
MODELS = {"whisper-tiny": ("hf-whisper", "openai/whisper-tiny"), "whisper-base": ("hf-whisper", "openai/whisper-base"),
          "moonshine-base": ("hf-moonshine", "UsefulSensors/moonshine-base")}
TRUTH_MODEL = "whisper-base"
FILES = {"clean_testset_wav.zip": "b60bfc2b9df38466d8eb45057c087c6e5dee196c67a0ed7bca628ae2a8611c4e",
         "noisy_testset_wav.zip": "ba4a893f45b627f15209ba30ab584dd495b3680cd6f216dd08d143a257611993",
         "testset_txt.zip": "b31edb40b39c8ddc99eda13ea038f51be3065a8b70272c8051ffc2c4294cf1d3",
         "logfiles.zip": "ab37c41f31bc2b6c204cc10c8e5bb8bde800307fdb49f277e366d986fefc9e39"}
BITSTREAMS = {"clean_testset_wav.zip": "dec213d3-bf57-4777-9663-c24bdce92d5e", "noisy_testset_wav.zip": "13c1bfbf-14a6-41db-9b41-8f7310f01ad5",
              "testset_txt.zip": "09e12592-879e-4749-bf6d-e34cf0d1ddf6", "logfiles.zip": "11185dc8-9cf1-405b-b858-35bd6a04aedd"}
API = "https://datashare.ed.ac.uk/server/api/core/bitstreams/{}/content"


def _pattern(rng, n):
    """Random clean/noisy bits with at least two noisy runs and two clean runs (so the noise comes and goes)."""
    while True:
        bits = rng.random(n) < 0.5
        runs = 1 + int((bits[1:] != bits[:-1]).sum())
        if bits.sum() >= 2 and (~bits).sum() >= 2 and runs >= 4:
            return bits


def _segments(rng, dur):
    """4 to 7 segments whose lengths are random in SEG_RANGE, scaled so that they tile the clip exactly."""
    n = int(np.clip(round(dur / np.mean(SEG_RANGE)), 4, 7))
    lengths = rng.uniform(*SEG_RANGE, size=n); lengths *= dur / lengths.sum()
    return [0.0] + [float(v) for v in np.cumsum(lengths)[:-1]] + [dur]


def _alternate(clean, noisy, segs):
    """Take each segment from clean or noisy, raised-cosine crossfade at the cuts."""
    n = len(clean); x = clean.copy(); mask = np.zeros(n, np.float32)
    for a, b, noisy_seg in segs:
        if noisy_seg:
            mask[int(a * SR):int(b * SR)] = 1.0
    k = int(XFADE_S * SR)
    if k > 1:
        w = np.hanning(2 * k)[:k]
        m = mask.copy()
        d = np.diff(mask, prepend=mask[0])
        for i in np.flatnonzero(d != 0):
            lo, hi = max(0, i - k // 2), min(n, i - k // 2 + k)
            ramp = w[: hi - lo] if d[i] > 0 else (1 - w)[: hi - lo]
            m[lo:hi] = ramp
        mask = m
    x = clean * (1 - mask) + noisy * mask
    return x.astype(np.float32), mask


def prepare():
    import librosa

    from ways.provenance import sha256

    dl = CACHE / "dl"; dl.mkdir(parents=True, exist_ok=True)
    for f, h in FILES.items():
        p = dl / f
        if not p.exists():
            import urllib.request
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(API.format(BITSTREAMS[f]), p)
        assert sha256(p) == h, f"{f}: sha256 differs from the one recorded here"
        import zipfile
        with zipfile.ZipFile(p) as z:
            z.extractall(CACHE)
    log = pd.read_csv(CACHE / "log_testset.txt", sep=r"\s+", header=None, names=["id", "noise", "snr_db"])
    txt = {p.stem: p.read_text().strip() for p in (CACHE / "testset_txt").glob("*.txt")}
    log["text"] = log.id.map(txt)
    log = log.dropna().sort_values("id").reset_index(drop=True)
    log["duration"] = [sf.info(CACHE / "clean_testset_wav" / f"{i}.wav").duration for i in log.id]
    pool = log[log.duration >= 2.0].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sel = pool.iloc[np.sort(rng.choice(len(pool), N_CLIPS, replace=False))].reset_index(drop=True)
    segs_all = {}
    for r in sel.itertuples():
        c, sr0 = sf.read(CACHE / "clean_testset_wav" / f"{r.id}.wav", dtype="float32")
        nz, sr1 = sf.read(CACHE / "noisy_testset_wav" / f"{r.id}.wav", dtype="float32")
        assert sr0 == sr1 and len(c) == len(nz)
        c = librosa.resample(c, orig_sr=sr0, target_sr=SR).astype(np.float32)
        nz = librosa.resample(nz, orig_sr=sr1, target_sr=SR).astype(np.float32)
        dur = len(c) / SR
        edges = _segments(rng, dur)
        bits = _pattern(rng, len(edges) - 1)
        segs = [(float(a), float(b), bool(s)) for a, b, s in zip(edges[:-1], edges[1:], bits, strict=True)]
        alt, mask = _alternate(c, nz, segs)
        sf.write(CACHE / f"{r.id}_clean.wav", c, SR); sf.write(CACHE / f"{r.id}_noisy.wav", nz, SR); sf.write(CACHE / f"{r.id}_alt.wav", alt, SR)
        segs_all[r.id] = segs
    sel["duration"] = [len(sf.read(CACHE / f"{i}_clean.wav", dtype="float32")[0]) / SR for i in sel.id]
    sel["segments"] = sel.id.map(lambda i: json.dumps(segs_all[i]))
    sel["n_segments"] = sel.id.map(lambda i: len(segs_all[i]))
    sel["noisy_share"] = sel.id.map(lambda i: sum(b - a for a, b, s in segs_all[i] if s) / sum(b - a for a, b, s in segs_all[i]))
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "manifest.csv", index=False)
    print(sel[["noise", "snr_db"]].value_counts().sort_index().to_string())
    print(f"{len(sel)} clips, duration {sel.duration.min():.1f} to {sel.duration.max():.1f} s, segments {sel.n_segments.min()} to {sel.n_segments.max()}, noisy share {sel.noisy_share.mean():.2f}")
    from ways.quality import Quality
    q = Quality("v3")
    det = []
    for r in sel.itertuples():
        row = dict(id=r.id, noise=r.noise, snr_db=r.snr_db)
        for cond in ("clean", "noisy", "alt"):
            wav, _ = sf.read(CACHE / f"{r.id}_{cond}.wav", dtype="float32")
            P = q.frames(wav); np.save(CACHE / f"{r.id}_aadd_{cond}.npy", P)
            fl = (P >= 0.5).any(1); t = np.arange(len(P)) * 0.02
            row[f"flagged_{cond}"] = float(fl.mean())
            if cond == "alt":
                inside = np.zeros(len(P), bool)
                for a, b, s in segs_all[r.id]:
                    if s:
                        inside |= (t >= a) & (t < b)
                row["alt_flagged_in_noisy"] = float(fl[inside].mean()); row["alt_flagged_in_clean"] = float(fl[~inside].mean())
                row["alt_n_spans_true"] = int(sum(1 for a, b, s in segs_all[r.id] if s))
                from ways.figure import degraded_spans
                row["alt_n_spans_aadd"] = len(degraded_spans(P, 0.02))
            if cond == "noisy":
                from ways.quality import CLASSES
                row["noisy_top_class"] = CLASSES[int(P.mean(0).argmax())]
        det.append(row)
    pd.DataFrame(det).to_csv(OUT / "aadd_detection.csv", index=False)
    d = pd.DataFrame(det)
    print(d[["flagged_clean", "flagged_noisy", "flagged_alt", "alt_flagged_in_noisy", "alt_flagged_in_clean"]].mean().round(3).to_string())
    print(d.groupby("snr_db")[["flagged_noisy", "alt_flagged_in_noisy", "alt_flagged_in_clean"]].mean().round(2).to_string())


def run(name, kind, model_id, max_minutes=30.0):
    from common import REVISIONS, resume_rows

    from ways import explain, load_backend
    from ways.metrics import wer

    man = pd.read_csv(OUT / "manifest.csv")
    d = OUT / name; (d / "npz").mkdir(parents=True, exist_ok=True)
    rows_f = d / "rows.jsonl"
    done = {(r["id"], r["cond"]) for r in resume_rows(rows_f, lambda r: (d / "npz" / f"{r['id']}_{r['cond']}.npz").exists())}
    be = load_backend(kind, model_id, revision=REVISIONS.get(model_id))
    t_run = time.time(); n_new = 0
    for r in man.itertuples():
        for cond in ("clean", "noisy", "alt"):
            if (r.id, cond) in done:
                continue
            if (time.time() - t_run) / 60 > max_minutes:
                print(f"{name}: time cap of {max_minutes:g} min reached after {n_new} new clip-conditions", flush=True)
                (d / "time_cap.json").write_text(json.dumps(dict(max_minutes=max_minutes, done=len(done) + n_new, total=3 * len(man))))
                sys.exit(f"{name}: stopped at the time cap with clip-conditions left. Run again to continue, or raise the cap.")
            t0 = time.time()
            wav, _ = sf.read(CACHE / f"{r.id}_{cond}.wav", dtype="float32")
            att = explain(be, wav, batch=64)
            att.save(d / "npz" / f"{r.id}_{cond}.npz")
            w = wer(r.text, att.text)
            row = dict(id=r.id, cond=cond, noise=r.noise, snr_db=float(r.snr_db), model=name, text=att.text, wer=w["wer"],
                       n_tokens=int(att.delta.shape[0]), mean_surprisal0=float(att.surprisal0.mean()), seconds=time.time() - t0)
            with open(rows_f, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_new += 1
            print(f"{name} {r.id} {cond} WER {w['wer']:.2f} {row['seconds']:.1f}s", flush=True)
    if n_new or not (d / "time_cap.json").exists():
        (d / "time_cap.json").write_text(json.dumps(dict(max_minutes=max_minutes, done=len(done) + n_new, total=3 * len(man), minutes=(time.time() - t_run) / 60)))
    from ways.provenance import write as write_manifest
    write_manifest(d / "provenance.json", inputs={**{f: str(CACHE / "dl" / f) for f in FILES}, "manifest.csv": str(OUT / "manifest.csv")},
                   seeds={"sample": SEED, "segments": SEED, "bootstrap": 8001}, models={model_id: REVISIONS.get(model_id)},
                   extra=dict(backend=be.describe(), patch_s=0.1, n_bands=8, n_clips=int(len(man)), conditions=["clean", "noisy", "alt"],
                              max_minutes=max_minutes, new_this_run=n_new))


def _norm_words(words):
    from ways.metrics import normalize_text
    return [normalize_text(w) for w in words]


def _word_profiles(att):
    """Per word: the mass profile over time [T] of its tokens, normalised to sum 1, and the word strings."""
    from ways.metrics import token_mass
    m = np.nan_to_num(token_mass(att)).sum(1)
    words = att.words if att.words else [dict(word=p.strip(), toks=[k]) for k, p in enumerate(att.pieces)]
    prof = np.stack([m[w["toks"]].sum(0) for w in words]) if words else np.zeros((0, att.n_t))
    tot = prof.sum(1, keepdims=True)
    prof = np.where(tot > 0, prof / np.maximum(tot, 1e-12), 0.0)
    return words, prof


def _seg_index(t, segs):
    for i, (a, b, _noisy) in enumerate(segs):
        if a <= t < b:
            return i
    return len(segs) - 1


def summarise():
    from ways.attribution import Attribution
    from ways.metrics import bootstrap_ci

    man = pd.read_csv(OUT / "manifest.csv").set_index("id")
    segs = {i: [tuple(s) for s in json.loads(v)] for i, v in man.segments.items()}
    aadd = pd.read_csv(OUT / "aadd_detection.csv")
    S = dict(design=dict(n_clips=int(len(man)), seed=SEED, source="VoiceBank-DEMAND test set (Edinburgh DataShare 10283/2791, CC BY 4.0), 2 speakers, resampled 48 to 16 kHz",
                         segments=dict(length_s=list(SEG_RANGE), crossfade_s=XFADE_S, rule="random clean/noisy bits, at least two noisy runs and two clean runs"),
                         noisy_share=float(man.noisy_share.mean()), n_segments=[int(man.n_segments.min()), int(man.n_segments.max())],
                         truth="word times from whisper-base on the clean recording (cross-attention timestamps), matched to each model's words by string alignment"),
             aadd=dict(flagged=aadd[["flagged_clean", "flagged_noisy", "flagged_alt", "alt_flagged_in_noisy", "alt_flagged_in_clean"]].mean().round(4).to_dict(),
                       by_snr=aadd.groupby("snr_db")[["flagged_noisy", "alt_flagged_in_noisy", "alt_flagged_in_clean"]].mean().round(3).to_dict("index"),
                       spans_true=float(aadd.alt_n_spans_true.mean()), spans_aadd=float(aadd.alt_n_spans_aadd.mean())),
             models={})
    truth = {}
    for cid in man.index:
        p = OUT / TRUTH_MODEL / "npz" / f"{cid}_clean.npz"
        if p.exists():
            a = Attribution.load(p)
            if a.words and "t0" in a.words[0]:
                truth[cid] = [(w["word"], 0.5 * (w["t0"] + w["t1"])) for w in a.words]
    if not truth:
        sys.exit(f"no saved maps of {TRUTH_MODEL} on the clean recordings under {OUT / TRUTH_MODEL / 'npz'}: "
                 "run the models first (experiments/run_demand.sh). Nothing was written.")
    word_rows, clip_rows = [], []
    for name in MODELS:
        d = OUT / name
        if not (d / "rows.jsonl").exists():
            continue
        rows = pd.DataFrame([json.loads(l) for l in open(d / "rows.jsonl")]).drop_duplicates(["id", "cond"], keep="last")
        M = dict(n_clips=int(rows.id.nunique()), wer={c: bootstrap_ci(rows[rows.cond == c].wer) for c in ("clean", "noisy", "alt")})
        if (d / "time_cap.json").exists():
            M["time_cap"] = json.loads((d / "time_cap.json").read_text())
        for cid, g in rows.groupby("id"):
            if cid not in truth or "alt" not in set(g.cond):
                continue
            sg = segs[cid]; dur = float(man.loc[cid, "duration"])
            att = Attribution.load(d / "npz" / f"{cid}_alt.npz")
            words, prof = _word_profiles(att)
            hyp = _norm_words([w["word"] for w in words]); ref = _norm_words([w for w, _ in truth[cid]])
            sm = SequenceMatcher(None, ref, hyp, autojunk=False)
            pairs = [(i + o, j + o) for i, j, n in sm.get_matching_blocks() for o in range(n)]
            noisy_share = float(man.loc[cid, "noisy_share"])
            cr = dict(model=name, id=cid, n_words=len(hyp), n_matched=len(pairs), noisy_share=noisy_share, n_segments=len(sg))
            hits = []
            for i, j in pairs:
                if prof[j].sum() <= 0:
                    continue
                t_true = truth[cid][i][1]
                si = _seg_index(t_true, sg); spoken_noisy = sg[si][2]
                top = int(prof[j].argmax()); t_top = (top + 0.5) * att.patch_s
                sj = _seg_index(t_top, sg)
                t0 = np.arange(att.n_t) * att.patch_s; t1 = t0 + att.patch_s
                in_noisy = sum(float((prof[j] * np.clip((np.minimum(t1, b) - np.maximum(t0, a)) / att.patch_s, 0, 1)).sum()) for a, b, s in sg if s)
                a, b = sg[si][0], sg[si][1]
                in_own = float((prof[j] * np.clip((np.minimum(t1, b) - np.maximum(t0, a)) / att.patch_s, 0, 1)).sum())
                wr = dict(model=name, id=cid, word=hyp[j], t_true=t_true, t_top=t_top, spoken_noisy=spoken_noisy, top_noisy=sg[sj][2],
                          same_segment=si == sj, seg_share=(b - a) / dur, mass_in_noisy=in_noisy, mass_in_own_segment=in_own, dist_s=abs(t_top - t_true))
                word_rows.append(wr); hits.append(wr)
            if hits:
                h = pd.DataFrame(hits)
                cr.update(n_scored=len(h), agree=float((h.spoken_noisy == h.top_noisy).mean()),
                          chance_agree=float((h.spoken_noisy * noisy_share + ~h.spoken_noisy * (1 - noisy_share)).mean()),
                          same_segment=float(h.same_segment.mean()), chance_same_segment=float(h.seg_share.mean()),
                          recall_noisy=float(h[h.spoken_noisy].top_noisy.mean()) if h.spoken_noisy.any() else np.nan,
                          spec_clean=float((~h[~h.spoken_noisy].top_noisy).mean()) if (~h.spoken_noisy).any() else np.nan,
                          mass_noisy_when_spoken_noisy=float(h[h.spoken_noisy].mass_in_noisy.mean()) if h.spoken_noisy.any() else np.nan,
                          mass_noisy_when_spoken_clean=float(h[~h.spoken_noisy].mass_in_noisy.mean()) if (~h.spoken_noisy).any() else np.nan,
                          mass_own_segment=float(h.mass_in_own_segment.mean()), dist_median=float(h.dist_s.median()))
            clip_rows.append(cr)
        C = pd.DataFrame([c for c in clip_rows if c["model"] == name])
        if len(C) and "agree" in C:
            for k in ("agree", "chance_agree", "same_segment", "chance_same_segment", "recall_noisy", "spec_clean",
                      "mass_noisy_when_spoken_noisy", "mass_noisy_when_spoken_clean", "mass_own_segment", "dist_median"):
                M[k] = bootstrap_ci(C[k])
            M["n_words_scored"] = int(C.n_scored.sum()); M["n_clips_scored"] = int((C.n_scored > 0).sum())
        S["models"][name] = M
    pd.DataFrame(word_rows).to_csv(OUT / "words.csv", index=False); pd.DataFrame(clip_rows).to_csv(OUT / "clips.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    ci = lambda d: f"{d['mean']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
    L = [f"**VoiceBank-DEMAND, {len(man)} clips, three conditions.** AADD v3 flags: clean {S['aadd']['flagged']['flagged_clean']:.2f}, "
         f"noisy {S['aadd']['flagged']['flagged_noisy']:.2f}; on the alternating clips {S['aadd']['flagged']['alt_flagged_in_noisy']:.2f} of noisy frames and "
         f"{S['aadd']['flagged']['alt_flagged_in_clean']:.2f} of clean frames; {S['aadd']['spans_true']:.1f} true noisy runs per clip, {S['aadd']['spans_aadd']:.1f} AADD spans.", "",
         "| model | WER clean | WER noisy | WER alternating | words placed on the right side (noisy or clean) | chance | word placed in its own segment | chance | mass in noisy audio, word spoken in noise | same, word spoken in clean |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, M in S["models"].items():
        if "agree" in M:
            L.append(f"| {name} | {ci(M['wer']['clean'])} | {ci(M['wer']['noisy'])} | {ci(M['wer']['alt'])} | {ci(M['agree'])} | {M['chance_agree']['mean']:.2f} | "
                     f"{ci(M['same_segment'])} | {M['chance_same_segment']['mean']:.2f} | {ci(M['mass_noisy_when_spoken_noisy'])} | {ci(M['mass_noisy_when_spoken_clean'])} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else 30.0)
    elif cmd == "summarise":
        summarise()
