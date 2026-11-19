"""Figures in the style of AADD's file inspector: a blue sequential ramp for audio quality only, a
yellow-to-orange spectrogram, and contrasting categorical colours that link each token to the
audio it used. Fonts are 12 pt or larger and every figure is about 10 x 8 in at 300 dpi."""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from .attribution import Attribution
from .figure import save_figure
from .metrics import token_mass
from .quality import CLASSES, SEG_S

BG, GRID, BORDER, TXT, TXT2, MUTED = "#fcfcfb", "#ecebe7", "#e4e3df", "#0b0b0b", "#52514e", "#77766f"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE = LinearSegmentedColormap.from_list("aadd_seq", SEQ)
CAT = ["#2a78d6", "#c2185b", "#1b9e77", "#7e3fc2"]
MEL_CMAP = "YlOrBr"
DPI = 300
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "figure.facecolor": BG,
                     "axes.facecolor": BG, "axes.edgecolor": BORDER, "xtick.color": TXT2, "ytick.color": TXT2,
                     "axes.labelcolor": TXT2, "svg.fonttype": "none"})


def col(k):
    return CAT[k % len(CAT)]


def logmel(wav: np.ndarray) -> np.ndarray:
    """Whisper-style log10 mel (80 slaney mels, 25 ms window, 10 ms hop, floor at max - 8)."""
    import librosa

    S = np.abs(librosa.stft(np.asarray(wav, np.float32), n_fft=400, hop_length=160, window="hann", center=True)) ** 2
    mel = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80) @ S[:, : len(wav) // 160]
    L = np.log10(np.maximum(mel, 1e-10))
    return np.maximum(L, L.max() - 8)


def _tidy(ax, dur, bottom=False):
    ax.set_xlim(0, dur)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(labelbottom=bottom, colors=TXT2)
    if bottom:
        ax.set_xlabel("time (s)")


def _label(ax, text):
    ax.set_title(text, loc="left", fontsize=12, color=TXT2, pad=4)


def _mel(ax, L):
    nf = L.shape[1]
    ax.imshow(L, origin="lower", aspect="auto", cmap=MEL_CMAP, extent=[0, nf * 0.01, 0, 80], interpolation="nearest",
              vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
    ax.set_ylim(0, 80); ax.set_yticks([0, 40, 80]); ax.set_yticklabels(["0", "1.9 k", "8 k"]); ax.set_ylabel("Hz (mel)")


DEG = "#0d366b"


def _flag_runs(probs, thr=0.5, min_gap=0.1):
    """Contiguous stretches (start_s, end_s) where any AADD class is at or above thr, gaps under min_gap merged."""
    if probs is None:
        return []
    fl = (np.asarray(probs) >= thr).any(1)
    runs, a = [], None
    for i, v in enumerate(fl):
        if v and a is None:
            a = i
        if not v and a is not None:
            runs.append([a * SEG_S, i * SEG_S]); a = None
    if a is not None:
        runs.append([a * SEG_S, len(fl) * SEG_S])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def _bands(ax, probs, top_bar=True):
    """Translucent blue bands over the degraded stretches, plus a solid blue rule along the top edge."""
    for a, b in _flag_runs(probs):
        ax.axvspan(a, b, color=DEG, alpha=0.16, lw=0, zorder=2)
        if top_bar:
            y0, y1 = ax.get_ylim()
            ax.add_patch(Rectangle((a, y1 - (y1 - y0) * 0.035), b - a, (y1 - y0) * 0.035, facecolor=DEG, edgecolor="none", zorder=3))


def _strip(ax, probs, dur):
    """AADD's maximum class probability per 20 ms as a filled line, with the 0.5 threshold and the flagged stretches."""
    if probs is None:
        ax.text(dur / 2, 0.5, "no AADD scan", ha="center", va="center", color=MUTED); ax.set_yticks([]); return
    pmax = np.clip(probs.max(1), 0, 1)
    t = np.arange(len(pmax)) * SEG_S
    ax.fill_between(t, 0, pmax, step="post", color=SEQ[-3], alpha=0.85, lw=0)
    ax.axhline(0.5, color=TXT2, lw=1, ls="--")
    for a, b in _flag_runs(probs):
        ax.axvspan(a, b, color=DEG, alpha=0.16, lw=0, zorder=0)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"], fontsize=9)


def _colorbar(fig, x, y, w=0.16, h=0.014, label="AADD degradation probability"):
    cax = fig.add_axes([x, y, w, h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=BLUE), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cax.tick_params(labelsize=10, colors=TXT2, length=2); cax.set_title(label, fontsize=10, color=TXT2, pad=2)


def token_rows(att: Attribution, probs=None):
    """One dict per token: top patch, and, when an AADD scan is given, whether that patch sits on degraded audio."""
    from .quality import flag_profile

    m = token_mass(att)
    f = flag_profile(probs, att.n_t, att.patch_s) if probs is not None else None
    rows = []
    for k in range(m.shape[0]):
        r = dict(k=k, tok=att.pieces[k])
        if not np.isnan(m[k]).any():
            prof = m[k].sum(0); bands = m[k].sum(1)
            r["top_t"], r["top_b"] = int(prof.argmax()), int(bands.argmax())
            r["patch"] = tuple(np.unravel_index(int(m[k].argmax()), m[k].shape))
            if f is not None:
                r["on_degraded"] = bool(f[r["top_t"]] >= 0.5)
        rows.append(r)
    return rows


def degraded_summary(rows, probs, duration):
    """'k of K words on degraded audio, expected e if no preference' from the rows and the clip's flagged share."""
    if probs is None:
        return ""
    n20 = int(np.ceil(duration / SEG_S))
    fl = (np.asarray(probs)[:n20] >= 0.5).any(1)
    frac = float(fl.mean()) if len(fl) else 0.0
    K = sum("patch" in r for r in rows); k = sum(r.get("on_degraded", False) for r in rows)
    if frac <= 0:
        return f"no degraded audio in this clip ({K} words)"
    if frac >= 1:
        return f"every frame degraded, so all {K} words sit on degraded audio"
    return f"{k} of {K} words on degraded audio, {frac * K:.0f} expected by chance ({frac:.0%} of the clip is degraded)"


def _lanes(rows, dur, patch_s, axis_in=8.1, char_in=0.095):
    lanes = []
    for r in rows:
        if "top_t" not in r:
            r["lane"] = 0; continue
        x0 = r["top_t"] * patch_s
        width = patch_s + (len(r["tok"].strip() or "_") + 3) * char_in * dur / axis_in + 0.02 * dur
        r["x0"], r["x1"] = x0, x0 + width
        for li in range(60):
            while len(lanes) <= li:
                lanes.append([])
            if all(r["x1"] <= a or r["x0"] >= b for a, b in lanes[li]):
                lanes[li].append((r["x0"], r["x1"])); r["lane"] = li; break
    return max(1, max(r.get("lane", 0) for r in rows) + 1)


def _place_labels(rows, att, dur, axis_in, mel_h=80, char_in=0.085, line_h=7.0):
    """Choose a text position for every token's label next to its box on the spectrogram so that no
    two labels overlap: try just above the box, just below, then further up or down in steps of one
    text line, then shift sideways. Returns [(k, x, y, va)] in data units."""
    B, ps = att.n_bands, att.patch_s
    mpb = mel_h // B
    placed, out = [], []
    cw = char_in * dur / axis_in
    for r in sorted(rows, key=lambda r: (r.get("top_t", 0), r["k"])):
        if "patch" not in r:
            continue
        b, t = r["patch"]
        text = f'{r["k"]} {r["tok"].strip() or "_"}'
        w = len(text) * cw + 0.2 * cw
        x0 = t * ps + ps / 2 - w / 2
        x0 = min(max(x0, 0.0), dur - w)
        cands = []
        for step in range(0, 6):
            cands.append((x0, (b + 1) * mpb + 0.5 + step * line_h, "bottom"))
            cands.append((x0, b * mpb - 0.5 - step * line_h, "top"))
        for dx in (0.6, -0.6, 1.2, -1.2):
            for step in range(0, 3):
                cands.append((min(max(x0 + dx * w, 0.0), dur - w), (b + 1) * mpb + 0.5 + step * line_h, "bottom"))
                cands.append((min(max(x0 + dx * w, 0.0), dur - w), b * mpb - 0.5 - step * line_h, "top"))
        for x, y, va in cands:
            y0, y1 = (y, y + line_h) if va == "bottom" else (y - line_h, y)
            if y0 < -0.5 or y1 > mel_h + 0.5:
                continue
            if all(x + w <= px or x >= px + pw or y1 <= py0 or y0 >= py1 for px, pw, py0, py1 in placed):
                placed.append((x, w, y0, y1)); out.append((r["k"], x, y, va)); break
        else:
            out.append((r["k"], x0, (b + 1) * mpb + 0.5, "bottom"))
    return out


def _sentence_lines(att, rows, axis_in, char_in=0.105, gap_in=0.22):
    """Lay the tokens out in reading order as a sentence, wrapping by the axis width (inches).
    Returns lines: list of [(k, x_in, text)] with x in inches from the left edge."""
    lines, x, cur = [], 0.0, []
    for r in rows:
        text = f'{r["k"]} {r["tok"].strip() or "_"}'
        w = 0.16 + len(text) * char_in + gap_in
        if cur and x + w > axis_in:
            lines.append(cur); cur, x = [], 0.0
        cur.append((r["k"], x, text)); x += w
    if cur:
        lines.append(cur)
    return lines


def _sentence_strip(ax, att, rows, axis_in):
    lines = _sentence_lines(att, rows, axis_in)
    ax.set_xlim(0, axis_in); ax.set_ylim(0, len(lines)); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for li, line in enumerate(lines):
        y = len(lines) - 1 - li + 0.5
        for k, x, text in line:
            ax.add_patch(Rectangle((x, y - 0.3), 0.16, 0.6, facecolor=col(k), edgecolor="white", lw=0.5))
            ax.text(x + 0.21, y, text, color=col(k), fontsize=12, ha="left", va="center", fontweight="bold")
            if rows[k].get("on_degraded"):
                ax.add_patch(Rectangle((x, y - 0.46), 0.16 + len(text) * 0.105 + 0.05, 0.07, facecolor=DEG, edgecolor="none"))
    return len(lines)


def _tokens_on_mel(axm, att, rows, dur, axis_in=8.1):
    B, ps = att.n_bands, att.patch_s
    mel_per_band = 80 // B
    for r in rows:
        if "patch" not in r:
            continue
        b, t = r["patch"]
        axm.add_patch(Rectangle((t * ps, b * mel_per_band), ps, mel_per_band, fill=False, edgecolor=col(r["k"]), lw=2.4 if r.get("on_degraded") else 2.0,
                                ls=(0, (2.5, 1.5)) if r.get("on_degraded") else "-", zorder=4))
    for k, x, y, va in _place_labels(rows, att, dur, axis_in):
        axm.text(x, y, f'{k} {att.pieces[k].strip() or "_"}', color=col(k), fontsize=10, ha="left", va=va, fontweight="bold",
                 zorder=4, clip_on=True, bbox=dict(boxstyle="square,pad=0.08", fc=BG, ec="none", alpha=0.75))


def clip_figure(att: Attribution, probs, wav, path, title="", window=None):
    """One model on one clip, through the universal figure (ways.figure.ways_figure)."""
    from .figure import from_attribution, ways_figure

    fig = ways_figure(wav, 16000, quality=probs, quality_hop=SEG_S, quality_classes=CLASSES,
                      models=[from_attribution(att, att.model)], window=window, title=title, out=path, style=dict(width_in=10.0))
    plt.close(fig)


def method_figure(att: Attribution, probs, wav, path, k: int | None = None, title=""):
    """The method on one clip and one token: the erasure grid, the erased patch, the surprisal change
    of every token for that patch, and the token's full map over the grid with the AADD flags."""
    import torch

    from . import erasure

    L = logmel(wav); dur = att.duration; ps, B = att.patch_s, att.n_bands
    mpb = 80 // B
    rows = token_rows(att)
    if k is None:
        cand = [(att.delta[r["k"]].max(), r["k"]) for r in rows if "patch" in r and 1 <= r["k"] < len(rows) - 1]
        k = max(cand)[1] if cand else 0
    b, t = rows[k]["patch"]
    grid = erasure.PatchGrid(len(wav), ps, B)
    xe = erasure.erase(torch.as_tensor(np.asarray(wav, np.float32)), grid, [(b, t)])[0].numpy()
    Le = logmel(xe)
    fig = plt.figure(figsize=(10, 8.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 2.2, 1.6, 0.9], width_ratios=[1, 1], hspace=0.62, wspace=0.22,
                          left=0.09, right=0.97, top=0.88, bottom=0.13)
    axa = fig.add_subplot(gs[0, :]); axb = fig.add_subplot(gs[1, 0]); axc = fig.add_subplot(gs[1, 1])
    axd = fig.add_subplot(gs[2, :]); axq = fig.add_subplot(gs[3, :])
    _mel(axa, L); _tidy(axa, dur)
    for tt in range(grid.n_t + 1):
        axa.axvline(tt * ps, color="white", lw=0.3, alpha=0.6)
    for bb in range(B + 1):
        axa.axhline(bb * mpb, color="white", lw=0.3, alpha=0.6)
    axa.add_patch(Rectangle((t * ps, b * mpb), ps, mpb, fill=False, edgecolor=col(k), lw=2.5, zorder=4))
    _label(axa, f"(a) Erasure grid: {ps * 1000:.0f} ms x {B} bands = {grid.n_t * B} patches. Highlighted: patch P = (band {b}, {t * ps:.1f}-{(t + 1) * ps:.1f} s)")
    f0, f1 = grid.frame_range(t); z0, z1 = max(0, f0 - 15), min(L.shape[1], f1 + 15)
    for ax, M, lab in ((axb, L, "(b) before"), (axc, Le, "(c) after erasing P")):
        ax.imshow(M[:, z0:z1], origin="lower", aspect="auto", cmap=MEL_CMAP, extent=[z0 * 0.01, z1 * 0.01, 0, 80], interpolation="nearest",
                  vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
        ax.add_patch(Rectangle((t * ps, b * mpb), ps, mpb, fill=False, edgecolor=col(k), lw=2.0))
        ax.set_yticks([0, 40, 80]); ax.set_yticklabels(["0", "1.9 k", "8 k"]); ax.set_xlabel("time (s)")
        drop = 10 * (L[b * mpb:(b + 1) * mpb, f0:f1].mean() - M[b * mpb:(b + 1) * mpb, f0:f1].mean())
        _label(ax, lab + (f": patch down {drop:.0f} dB" if ax is axc else ", log-mel input"))
    d = att.delta[:, b, t]
    K = len(d)
    axd.bar(np.arange(K), d, color=[col(k) if i == k else "#c9c7c1" for i in range(K)], width=0.8)
    axd.axhline(0, color=BORDER, lw=0.8)
    axd.set_xticks(range(K)); axd.set_xticklabels([p.strip() or "_" for p in att.pieces], rotation=90, fontsize=max(7, min(11, 320 / max(K, 1))))
    axd.set_ylabel("delta (nats)")
    _label(axd, f"(d) delta_k(P) = S_k(x\\P) - S_k(x) for every token k; token {k} '{att.pieces[k].strip()}' loses most")
    for s in axd.spines.values():
        s.set_color(BORDER)
    M = np.clip(att.delta[k], 0, None)
    axq.imshow(M, origin="lower", aspect="auto", cmap="Greys", extent=[0, grid.n_t * ps, 0, B], interpolation="nearest", vmin=0, vmax=max(M.max(), 1e-6))
    axq.add_patch(Rectangle((t * ps, b), ps, 1, fill=False, edgecolor=col(k), lw=2.0))
    axq.set_yticks([0.5, B - 0.5]); axq.set_yticklabels(["low", "high"]); axq.set_ylabel("band")
    _tidy(axq, dur, bottom=True)
    if probs is not None:
        fl = (probs >= 0.5).any(1)
        tt = np.arange(len(fl)) * SEG_S
        axq2 = axq.twinx(); axq2.set_ylim(0, 1); axq2.set_yticks([])
        axq2.bar(tt[fl], np.full(fl.sum(), 0.08), width=SEG_S, bottom=0.92, color=SEQ[-2], align="edge", lw=0)
    _label(axq, f"(e) token {k}'s map, max(delta, 0) over all patches; top bar = AADD-flagged frames (p >= 0.5)")
    fig.suptitle(title, x=0.09, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.975)
    fig.text(0.09, 0.012, "Erase one patch of the spectrogram, re-score the model's own transcript with the earlier tokens held fixed, "
             "record how much each token's surprisal rises. Repeat for every patch.", fontsize=10, color=TXT2, va="bottom", wrap=True)
    axq.set_xlabel("time (s)", labelpad=2)
    save_figure(fig, path, DPI); plt.close(fig)
    return k, (b, t)


def three_models_figure(atts: dict[str, Attribution], probs, wav, path, title="", wers: dict | None = None, window=None):
    """Same clip, one row per model, through the universal figure (ways.figure.ways_figure)."""
    from .figure import from_attribution, ways_figure

    models = [from_attribution(a, n, (wers or {}).get(n)) for n, a in atts.items()]
    fig = ways_figure(wav, 16000, quality=probs, quality_hop=SEG_S, quality_classes=CLASSES, models=models,
                      window=window, title=title, out=path)
    plt.close(fig)


def comparison_figure(clips: dict, tokens: dict, path, title="", band_edges=None):
    """Per model: WER, noise-focus ratio per clip, and the band shares. clips/tokens: {name: DataFrame}."""
    names = list(clips)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw=dict(left=0.06, right=0.98, top=0.82, bottom=0.2, wspace=0.36))
    ax = axes[0]
    for i, n in enumerate(names):
        w = clips[n]["wer"].dropna().clip(upper=1.5)
        ax.boxplot(w, positions=[i], widths=0.55, patch_artist=True, boxprops=dict(facecolor=col(i), alpha=0.35, edgecolor=col(i)),
                   medianprops=dict(color=col(i), lw=2), whiskerprops=dict(color=col(i)), capprops=dict(color=col(i)),
                   flierprops=dict(marker=".", markeredgecolor=col(i), alpha=0.5))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15); ax.set_ylabel("word error rate per clip (capped at 1.5)")
    _label(ax, "(a) How many words each model gets wrong")
    ax = axes[1]
    for i, n in enumerate(names):
        c = clips[n]
        f = c.loc[(c.frac_flagged >= 0.05) & (c.frac_flagged <= 0.95), "focus_ratio_mean"].dropna()
        ax.boxplot(f, positions=[i], widths=0.55, patch_artist=True, boxprops=dict(facecolor=col(i), alpha=0.35, edgecolor=col(i)),
                   medianprops=dict(color=col(i), lw=2), whiskerprops=dict(color=col(i)), capprops=dict(color=col(i)),
                   flierprops=dict(marker=".", markeredgecolor=col(i), alpha=0.5))
    ax.axhline(1, color=TXT2, lw=1, ls="--"); ax.text(len(names) - 0.5, 1.02, "no preference", ha="right", va="bottom", fontsize=10, color=TXT2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15); ax.set_ylabel("evidence on degraded audio / share of audio degraded")
    _label(ax, f"(b) Does a model lean on degraded audio? ({len(f)} clips)")
    ax = axes[2]
    B = 8
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        bs = [tokens[n][f"band{b}"].mean() for b in range(B)]
        ax.bar(np.arange(B) + (i - (len(names) - 1) / 2) * w, bs, width=w, color=col(i), label=n)
    ax.set_xticks(range(B))
    if band_edges is not None:
        ax.set_xticklabels([f"{band_edges[b] / 1000:.1f}" for b in range(B)]); ax.set_xlabel("band lower edge (kHz)")
    ax.set_ylabel("share of a word's evidence")
    _label(ax, "(c) Which pitch range the evidence comes from")
    for a in axes:
        for s in a.spines.values():
            s.set_color(BORDER)
    axes[2].legend(frameon=False, title="speech-to-text model", title_fontsize=11)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.97)
    save_figure(fig, path, DPI); plt.close(fig)
