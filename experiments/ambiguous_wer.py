"""Why the word error rate moved on the ambiguous clip: the clean and the degraded transcript of each
model aligned to the gold, every word placed on the audio where its erasure evidence sits. Reads the controlled experiment's saved attributions (no model run).

Usage: python experiments/ambiguous_wer.py
Writes figures/controlled_noise_wer__<clip>.png (+ svg) and experiments/results/ambiguous/wer_words.csv, wer_summary.md
"""
import json
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ways.attribution import Attribution
from ways.figure import FigureStyle, _rc, logmel, text_width_pt, wrap_to_width
from ways.metrics import normalize_text, token_mass

HERE = Path(__file__).resolve().parent
CTRL = HERE / "results/controlled"
OUT = HERE / "results/ambiguous"
FIG = HERE.parent / "figures"
CLIP = "2094-142345-0024"
MODELS = ["whisper-tiny", "whisper-base", "moonshine-base"]
S = FigureStyle()
OK, BAD = "#1b9e77", "#c2185b"


def words_of(att: Attribution):
    """Tokens merged into words: text, mean erasure profile over the word's tokens [patches], top-patch time."""
    m = token_mass(att); prof = np.where(np.isnan(m).all((1, 2))[:, None], np.nan, m.sum(1))
    words, cur = [], None
    for k, piece in enumerate(att.pieces):
        if piece.startswith(" ") or cur is None:
            cur = dict(text="", ks=[]); words.append(cur)
        cur["text"] += piece; cur["ks"].append(k)
    out = []
    for w in words:
        p = np.nanmean(prof[w["ks"]], 0) if np.isfinite(prof[w["ks"]]).any() else np.full(prof.shape[1], np.nan)
        norm = normalize_text(w["text"])
        if not norm:                         # a punctuation-only token hangs on the previous word
            if out:
                out[-1]["ks"] += w["ks"]
            continue
        out.append(dict(text=w["text"].strip(), norm=norm, ks=w["ks"], prof=p,
                        t=(int(np.nanargmax(p)) + 0.5) * att.patch_s if np.isfinite(p).any() else np.nan))
    return out


def align(gold: str, words):
    """Per hypothesis word: 'hit', 'sub' or 'ins' against the gold, from jiwer's alignment; plus deletions."""
    hyp = " ".join(w["norm"] for w in words)
    o = jiwer.process_words(normalize_text(gold), hyp)
    status, dels = {}, 0
    for ch in o.alignments[0]:
        for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
            status[j] = {"equal": "hit", "substitute": "sub", "insert": "ins"}[ch.type]
        if ch.type == "delete":
            dels += ch.ref_end_idx - ch.ref_start_idx
    return [status.get(j, "ins") for j in range(len(words))], dels, o


def main():
    man = pd.read_csv(CTRL / "manifest.csv").set_index("id")
    a, b = float(man.loc[CLIP, "win_start"]), float(man.loc[CLIP, "win_end"]); gold = man.loc[CLIP, "text"]
    import soundfile as sf
    wav, sr = sf.read(HERE / "cache/controlled" / f"{CLIP}_deg.wav", dtype="float32"); dur = len(wav) / sr
    L = logmel(wav, sr)
    rows, summ = [], []
    _rc(S)
    fig_h = 1.0 + 1.6 + len(MODELS) * 2.1 + 1.1
    fig = plt.figure(figsize=(S.width_in, fig_h))
    left, right = 0.085, 0.985
    axis_pt = (right - left) * S.width_in * 72; per_pt = dur / axis_pt
    gs = fig.add_gridspec(1 + len(MODELS), 1, height_ratios=[1.6] + [2.1] * len(MODELS), hspace=0.5, left=left, right=right,
                          top=1 - 1.0 / fig_h, bottom=1.1 / fig_h)
    axm = fig.add_subplot(gs[0])
    axm.imshow(L, origin="lower", aspect="auto", cmap=S.mel_cmap, extent=[0, L.shape[1] * 0.01, 0, 80], interpolation="nearest",
               vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
    axm.axvspan(a, b, color=S.degraded, alpha=S.band_alpha, lw=0); axm.set_xlim(0, dur); axm.set_yticks([0, 40, 80]); axm.set_yticklabels(["0", "1.9 k", "8 k"])
    axm.set_ylabel("Hz (mel)"); axm.tick_params(labelbottom=False, colors=S.text_soft)
    axm.set_title(f"the degraded recording: pink noise at 0 dB over {a:.1f}-{b:.1f} s (blue)", loc="left", fontsize=S.font_size, color=S.text_soft, pad=4)
    for sp in axm.spines.values():
        sp.set_color(S.border)
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[1 + i])
        per = {}
        for cond in ("clean", "deg"):
            att = Attribution.load(CTRL / model / "npz" / f"{CLIP}_{cond}.npz")
            words = words_of(att); status, dels, o = align(gold, words)
            per[cond] = dict(words=words, status=status, wer=(o.substitutions + o.deletions + o.insertions) / len(normalize_text(gold).split()),
                             dels=dels, text=att.text)
            for w, st in zip(words, status, strict=True):
                inwin = a <= w["t"] <= b if np.isfinite(w["t"]) else False
                mass_in = float(np.nansum(w["prof"][int(a / att.patch_s): int(np.ceil(b / att.patch_s))])) if np.isfinite(w["prof"]).any() else np.nan
                rows.append(dict(model=model, cond=cond, word=w["text"], norm=w["norm"], status=st, t=w["t"], in_window=inwin, mass_in_window=mass_in))
        # words both transcripts share (same normalised word, nearest in time), counted for the table
        used = set()
        for wc in per["clean"]["words"]:
            cands = [(abs(wd["t"] - wc["t"]), j) for j, wd in enumerate(per["deg"]["words"]) if wd["norm"] == wc["norm"] and j not in used]
            if cands and min(cands)[0] <= 1.0:
                used.add(min(cands)[1])
        for cond, y, sign in (("clean", 1.0, 1), ("deg", 0.0, -1)):
            p = per[cond]
            rows_end = [-1.0, -1.0, -1.0]                       # right edge of the last label in each label row
            for w, st in sorted(zip(p["words"], p["status"], strict=True), key=lambda r: r[0]["t"]):
                if not np.isfinite(w["t"]):
                    continue
                col = OK if st == "hit" else BAD
                ax.add_patch(Rectangle((w["t"] - 0.05, y - 0.075), 0.1, 0.15, facecolor=col if st == "hit" else "white", edgecolor=col, lw=1.4, zorder=3))
                wd = (text_width_pt(w["text"], S.label_size - 1, "bold", S.font) + 6) * per_pt
                x0 = min(max(w["t"] - wd / 2, 0.0), dur - wd)
                row = next((r for r, e in enumerate(rows_end) if x0 > e), 2)
                rows_end[row] = x0 + wd
                ly = y + sign * (0.13 + 0.2 * row)
                ax.text(w["t"] if 0 <= w["t"] - wd / 2 and w["t"] + wd / 2 <= dur else x0 + wd / 2, ly, w["text"], ha="center",
                        va="bottom" if sign > 0 else "top", fontsize=S.label_size - 1, color=col, fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor=S.background, edgecolor="none", alpha=0.85))
        ax.axvspan(a, b, color=S.degraded, alpha=S.band_alpha, lw=0, zorder=0)
        ax.set_xlim(0, dur); ax.set_ylim(-0.8, 1.8); ax.set_yticks([0.0, 1.0]); ax.set_yticklabels(["degraded", "clean"], fontsize=S.label_size)
        ax.tick_params(labelbottom=(i == len(MODELS) - 1), colors=S.text_soft)
        if i == len(MODELS) - 1:
            ax.set_xlabel("time (s), each word at the patch its erasure evidence ranks first", labelpad=2)
        for sp in ax.spines.values():
            sp.set_color(S.border)
        d = pd.DataFrame([r for r in rows if r["model"] == model])
        n_in = {c: int(((d.cond == c) & d.in_window).sum()) for c in ("clean", "deg")}
        hit_in = {c: int(((d.cond == c) & d.in_window & (d.status == "hit")).sum()) for c in ("clean", "deg")}
        hit_out = {c: int(((d.cond == c) & ~d.in_window & (d.status == "hit")).sum()) for c in ("clean", "deg")}
        n_out = {c: int(((d.cond == c) & ~d.in_window).sum()) for c in ("clean", "deg")}
        shared = len(used)
        m_in = {c: float(d[(d.cond == c) & d.in_window].mass_in_window.mean()) for c in ("clean", "deg")}
        m_out = {c: float(d[(d.cond == c) & ~d.in_window].mass_in_window.mean()) for c in ("clean", "deg")}
        summ.append(dict(model=model, wer_clean=per["clean"]["wer"], wer_deg=per["deg"]["wer"], text_clean=per["clean"]["text"], text_deg=per["deg"]["text"],
                         words_in_window=f"{n_in['clean']} -> {n_in['deg']}", correct_in_window=f"{hit_in['clean']} of {n_in['clean']} -> {hit_in['deg']} of {n_in['deg']}",
                         correct_outside=f"{hit_out['clean']} of {n_out['clean']} -> {hit_out['deg']} of {n_out['deg']}", shared_words=shared,
                         mass_in_window_of_window_words=f"{m_in['clean']:.2f} -> {m_in['deg']:.2f}", mass_in_window_of_outside_words=f"{m_out['clean']:.2f} -> {m_out['deg']:.2f}"))
        ttl = (f"{model}: WER {per['clean']['wer']:.0%} clean, {per['deg']['wer']:.0%} degraded. Words on the noisy stretch "
               f"{hit_in['clean']} of {n_in['clean']} right on the clean recording, {hit_in['deg']} of {n_in['deg']} on the degraded one; "
               f"outside it {hit_out['clean']} of {n_out['clean']} and {hit_out['deg']} of {n_out['deg']}.")
        ax.set_title(wrap_to_width(ttl, axis_pt, S.font_size), loc="left", fontsize=S.font_size, color=S.text_soft, pad=4)
    fig.suptitle(wrap_to_width(f"The same clip clean and degraded through three models, each word aligned to the gold\ngold: '{gold}'",
                               axis_pt, S.font_size + 1, "bold", S.font), x=left, ha="left", fontsize=S.font_size + 1, fontweight="bold", color=S.text, y=0.995, va="top")
    fig.text(left, 0.008, wrap_to_width("Green filled = the word matches the gold at its place in the sentence, red outline = a substitution or an "
             "insertion. Blue band = the noisy two seconds.", axis_pt, S.label_size, font=S.font),
             fontsize=S.label_size, color=S.text_soft, va="bottom")
    out = FIG / f"controlled_noise_wer__{CLIP}.png"
    fig.savefig(out, dpi=S.dpi, facecolor=S.background); fig.savefig(out.with_suffix(".svg"), facecolor=S.background); plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "wer_words.csv", index=False)
    lines = ["| model | WER clean -> degraded | words on the noisy stretch | right on it | right outside it | shared words | evidence in the noise, window words | evidence in the noise, outside words |",
             "|---|---|---|---|---|---|---|---|"]
    for s_ in summ:
        lines.append(f"| {s_['model']} | {s_['wer_clean']:.0%} -> {s_['wer_deg']:.0%} | {s_['words_in_window']} | {s_['correct_in_window']} | {s_['correct_outside']} | "
                     f"{s_['shared_words']} | {s_['mass_in_window_of_window_words']} | {s_['mass_in_window_of_outside_words']} |")
    (OUT / "wer_summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "wer_summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(lines)); print(out)


if __name__ == "__main__":
    main()
