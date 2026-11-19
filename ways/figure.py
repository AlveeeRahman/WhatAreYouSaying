"""One figure, same structure every time: which audio each word used, and whether that audio was clean.

This file is self-contained (numpy and matplotlib only) so it can be copied into any Python project.
Give it a waveform, optionally a degradation score per frame from any detector, and optionally one
or more sets of word attributions from any method, and it draws four lanes with one time axis:

  1. Damage: one row per kind of degradation that reaches the threshold, plus the maximum, as a blue
     heat map over time (dark = sure). Any stretch above the threshold is a "degraded span".
  2. Spectrogram of the audio (orange = loud, pale = quiet) with the degraded spans painted as blue
     bands, and, per model, one box per word on the patch of audio that mattered most to it.
  3. The transcript as one sentence in the box colours; a word on degraded audio is underlined.
  4. The verdict in each model's row title: k of K words on degraded audio, e expected by chance.

Minimal use, with your own detector and your own attribution method:

    from ways.figure import ways_figure
    ways_figure(wav, 16000,
                quality=probs, quality_hop=0.02, quality_classes=["noise", "reverb", ...],
                models=[{"name": "my-asr", "tokens": ["hello", "world"], "top": [(2, 14), (3, 20)],
                         "patch_s": 0.1, "n_bands": 8, "wer": 0.1}],
                title="my clip", out="my_clip.png")

`top` is, per token, (band index, time index) of the patch it used most; bands are equal groups of
80 mel bins from low to high, time indices are patches of `patch_s` seconds. `quality` is any
[n_frames, n_classes] array of probabilities at `quality_hop` seconds per frame (AADD gives 12
classes every 20 ms; a single-column array works too). Everything else is optional.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

BG, GRID, BORDER, TXT, TXT2, MUTED = "#fcfcfb", "#ecebe7", "#e4e3df", "#0b0b0b", "#52514e", "#77766f"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE = LinearSegmentedColormap.from_list("ways_seq", SEQ)
DEG = "#0d366b"
CAT = ["#2a78d6", "#c2185b", "#1b9e77", "#7e3fc2"]        # token colours: colour-blind safe adjacent pairs
MEL_CMAP = "YlOrBr"
DPI = 300
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.edgecolor": BORDER, "xtick.color": TXT2, "ytick.color": TXT2, "axes.labelcolor": TXT2,
                     "svg.fonttype": "none"})


def col(k):
    return CAT[k % len(CAT)]


# ----------------------------------------------------------------------------- audio -> log-mel (numpy only)
def _slaney_mel_edges(n_mels, fmin, fmax):
    """The n_mels + 2 edge frequencies of a slaney mel scale (linear to 1 kHz, log above), as librosa."""
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27
    hz_to_mel = lambda f: np.where(f >= min_log_hz, 15 + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, f / f_sp)
    mel_to_hz = lambda m: np.where(m >= 15, min_log_hz * np.exp(logstep * (m - 15)), f_sp * m)
    return mel_to_hz(np.linspace(hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax)), n_mels + 2))


def logmel(wav, sr=16000, n_fft=400, hop=160, n_mels=80, fmax=None):
    """Whisper-style log10 mel spectrogram: 25 ms Hann window, 10 ms hop, 80 slaney-mel bands, floor at max - 8."""
    wav = np.asarray(wav, np.float32)
    if sr != 16000:
        n_fft, hop = int(round(0.025 * sr)), int(round(0.010 * sr))
    pad = n_fft // 2
    x = np.pad(wav, (pad, pad), mode="reflect")
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(n_fft + 1)[:-1]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2                       # [n, n_fft//2+1]
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = _slaney_mel_edges(n_mels, 0.0, fmax or sr / 2)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.maximum(0, np.minimum((freqs - lo) / max(c - lo, 1e-9), (hi - freqs) / max(hi - c, 1e-9)))
        fb[m] *= 2.0 / (hi - lo)                                            # slaney normalisation
    L = np.log10(np.maximum(fb @ power.T, 1e-10))[:, : len(wav) // hop]
    return np.maximum(L, L.max() - 8)


# ----------------------------------------------------------------------------- degraded spans
def degraded_spans(quality, hop, threshold=0.5, min_gap=0.1):
    """(start_s, end_s) stretches where any class is at or above threshold; gaps under min_gap merged."""
    if quality is None:
        return []
    q = np.asarray(quality, np.float32)
    q = q[:, None] if q.ndim == 1 else q
    fl = (q >= threshold).any(1)
    runs, a = [], None
    for i, v in enumerate(fl):
        if v and a is None:
            a = i
        if not v and a is not None:
            runs.append([a * hop, i * hop]); a = None
    if a is not None:
        runs.append([a * hop, len(fl) * hop])
    out = []
    for r in runs:
        if out and r[0] - out[-1][1] < min_gap:
            out[-1][1] = r[1]
        else:
            out.append(r)
    return [tuple(r) for r in out]


def _flag_per_patch(quality, hop, threshold, patch_s, n_t):
    """Share of each patch's frames that are flagged -> f[n_t]; None without quality. Same rule as ways.quality.flag_profile."""
    if quality is None:
        return None
    q = np.asarray(quality, np.float32)
    q = q[:, None] if q.ndim == 1 else q
    fl = (q >= threshold).any(1).astype(np.float32)
    per = max(1, int(round(patch_s / hop)))
    f = np.zeros(n_t, np.float32)
    for t in range(n_t):
        seg = fl[t * per:(t + 1) * per]
        f[t] = seg.mean() if len(seg) else 0.0
    return f


# ----------------------------------------------------------------------------- layout helpers
def _tidy(ax, dur, bottom=False):
    ax.set_xlim(0, dur)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(labelbottom=bottom, colors=TXT2)
    if bottom:
        ax.set_xlabel("time (s)")


def _label(ax, text):
    ax.set_title(text, loc="left", fontsize=12, color=TXT2, pad=4)


def _bands(ax, spans):
    for a, b in spans:
        ax.axvspan(a, b, color=DEG, alpha=0.16, lw=0, zorder=2)
        y0, y1 = ax.get_ylim()
        ax.add_patch(Rectangle((a, y1 - (y1 - y0) * 0.035), b - a, (y1 - y0) * 0.035, facecolor=DEG, edgecolor="none", zorder=3))


def _damage_lane(ax, quality, hop, classes, threshold, dur, spans):
    """Rows: every class that reaches the threshold somewhere, then 'max'. Blue = probability."""
    q = np.asarray(quality, np.float32)
    q = q[:, None] if q.ndim == 1 else q
    classes = list(classes) if classes is not None else ([f"class {i}" for i in range(q.shape[1])] if q.shape[1] > 1 else ["degradation"])
    active = [j for j in range(q.shape[1]) if q[:, j].max() >= threshold]
    rows = [(classes[j], q[:, j]) for j in active]
    if q.shape[1] > 1:
        rows.append(("max", q.max(1)))
    elif not rows:
        rows = [(classes[0], q[:, 0])]
    H = np.stack([r for _, r in rows])
    ax.imshow(np.clip(H, 0, 1), aspect="auto", cmap=BLUE, extent=[0, H.shape[1] * hop, len(rows), 0], interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([n.replace("_", " ") for n, _ in rows], fontsize=10)
    for a, b in spans:
        ax.add_patch(Rectangle((a, 0), b - a, len(rows), fill=False, edgecolor=DEG, lw=1.2, zorder=3))
    if not active:
        ax.text(dur / 2, len(rows) / 2, f"nothing reaches {threshold:g}: clean", ha="center", va="center", color=MUTED, fontsize=11)


def _place_labels(items, dur, axis_in, mel_h=80, char_in=0.085, line_h=7.0):
    """Non-overlapping text positions beside the boxes. items: [(k, text, b, t, patch_s, mpb)]."""
    placed, out = [], []
    cw = char_in * dur / axis_in
    for k, text, b, t, ps, mpb in sorted(items, key=lambda r: (r[3], r[0])):
        w = (len(text) + 0.2) * cw
        x0 = min(max(t * ps + ps / 2 - w / 2, 0.0), max(dur - w, 0.0))
        cands = []
        for step in range(6):
            cands.append((x0, (b + 1) * mpb + 0.5 + step * line_h, "bottom"))
            cands.append((x0, b * mpb - 0.5 - step * line_h, "top"))
        for dx in (0.6, -0.6, 1.2, -1.2):
            for step in range(3):
                xx = min(max(x0 + dx * w, 0.0), max(dur - w, 0.0))
                cands.append((xx, (b + 1) * mpb + 0.5 + step * line_h, "bottom"))
                cands.append((xx, b * mpb - 0.5 - step * line_h, "top"))
        for x, y, va in cands:
            y0, y1 = (y, y + line_h) if va == "bottom" else (y - line_h, y)
            if y0 < -0.5 or y1 > mel_h + 0.5:
                continue
            if all(x + w <= px or x >= px + pw or y1 <= py0 or y0 >= py1 for px, pw, py0, py1 in placed):
                placed.append((x, w, y0, y1)); out.append((k, text, x, y, va)); break
        else:
            out.append((k, text, x0, (b + 1) * mpb + 0.5, "bottom"))
    return out


def _sentence_lines(tokens, axis_in, char_in=0.105, gap_in=0.22):
    lines, x, cur = [], 0.0, []
    for k, tok in enumerate(tokens):
        text = f"{k} {tok.strip() or '_'}"
        w = 0.16 + len(text) * char_in + gap_in
        if cur and x + w > axis_in:
            lines.append(cur); cur, x = [], 0.0
        cur.append((k, x, text)); x += w
    if cur:
        lines.append(cur)
    return lines


def _sentence_strip(ax, tokens, on_degraded, axis_in):
    lines = _sentence_lines(tokens, axis_in)
    ax.set_xlim(0, axis_in); ax.set_ylim(-0.15, len(lines) + 0.15); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for li, line in enumerate(lines):
        y = len(lines) - 1 - li + 0.5
        for k, x, text in line:
            ax.add_patch(Rectangle((x, y - 0.3), 0.16, 0.6, facecolor=col(k), edgecolor="white", lw=0.5))
            ax.text(x + 0.21, y, text, color=col(k), fontsize=12, ha="left", va="center", fontweight="bold")
            if on_degraded is not None and on_degraded[k]:
                ax.add_patch(Rectangle((x, y - 0.46), 0.16 + len(text) * 0.105 + 0.05, 0.07, facecolor=DEG, edgecolor="none"))
    return len(lines)


def verdict(on_degraded, frac):
    K = int(sum(v is not None for v in on_degraded)); k = int(sum(bool(v) for v in on_degraded))
    if frac is None:
        return f"{K} words"
    if frac <= 0:
        return f"no degraded audio in this clip ({K} words)"
    if frac >= 1:
        return f"every frame degraded, so all {K} words sit on degraded audio"
    return f"{k} of {K} words on degraded audio, {frac * K:.0f} expected by chance ({frac:.0%} of the clip is degraded)"


# ----------------------------------------------------------------------------- the figure
def ways_figure(wav, sr=16000, quality=None, quality_hop=0.02, quality_classes=None, threshold=0.5,
                models=None, window=None, title="", out=None, dpi=DPI, width_in=12.0):
    """Draw the standard WAYS figure. Returns the matplotlib Figure (saved to `out` and its .svg twin if given).

    models: list of dicts, each with
      name (str), tokens (list of str), top (list of (band, time) or None per token),
      patch_s (seconds per time index, default 0.1), n_bands (default 8), wer (optional float).
    window: optional (start_s, end_s) to mark a known damaged span with dashed lines.
    """
    import textwrap

    wav = np.asarray(wav, np.float32)
    dur = len(wav) / sr
    L = logmel(wav, sr)
    spans = degraded_spans(quality, quality_hop, threshold)
    frac = None
    if quality is not None:
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        n = int(np.ceil(dur / quality_hop)); fl = (q[:n] >= threshold).any(1)
        frac = float(fl.mean()) if len(fl) else 0.0
    models = models or []
    axis_in = width_in * (0.97 - 0.12)
    # per-model bookkeeping
    prepared = []
    for m in models:
        ps, B = m.get("patch_s", 0.1), m.get("n_bands", 8)
        n_t = int(np.ceil(dur / ps))
        f = _flag_per_patch(quality, quality_hop, threshold, ps, n_t)
        on = [None if tp is None else (bool(f[min(tp[1], n_t - 1)] >= 0.5) if f is not None else False) for tp in m["top"]]
        prepared.append((m, on, len(_sentence_lines(m["tokens"], axis_in))))
    n_dmg_rows = 1
    if quality is not None:
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        n_dmg_rows = max(1, sum(int(q[:, j].max() >= threshold) for j in range(q.shape[1])) + (1 if q.shape[1] > 1 else 0))
    hs = [0.22 * n_dmg_rows + 0.15]
    for _, _, nl in prepared:
        hs += [2.6, 0.42 * nl + 0.1]
    if not prepared:
        hs += [2.6]
    fig = plt.figure(figsize=(width_in, min(1.6 + sum(hs) * 0.62, 15)))
    gs = fig.add_gridspec(len(hs), 1, height_ratios=hs, hspace=0.3, left=0.12, right=0.97, top=0.89, bottom=0.07)
    # lane 1: damage
    axq = fig.add_subplot(gs[0])
    if quality is not None:
        _damage_lane(axq, quality, quality_hop, quality_classes, threshold, dur, spans)
        _label(axq, f"Damage per {quality_hop * 1000:.0f} ms frame (dark blue = sure); boxed stretches reach {threshold:g} and count as degraded")
    else:
        axq.text(dur / 2, 0.5, "no degradation scores given", ha="center", va="center", color=MUTED); axq.set_yticks([])
        _label(axq, "Damage: none measured")
    _tidy(axq, dur)
    if window is not None:
        axq.axvspan(window[0], window[1], color=DEG, alpha=0.10, lw=0, zorder=1)
    # lanes 2+3 per model (or just the spectrogram)
    if not prepared:
        axm = fig.add_subplot(gs[1]); _mel(axm, L, dur); _bands(axm, spans); _tidy(axm, dur, bottom=True)
        _label(axm, "Spectrogram (orange = loud, pale = quiet); blue bands = degraded audio")
    for i, (m, on, nl) in enumerate(prepared):
        ps, B = m.get("patch_s", 0.1), m.get("n_bands", 8); mpb = 80 // B
        axm = fig.add_subplot(gs[1 + 2 * i]); axt = fig.add_subplot(gs[2 + 2 * i])
        _mel(axm, L, dur); _bands(axm, spans)
        items = []
        for k, tp in enumerate(m["top"]):
            if tp is None:
                continue
            b, t = tp
            axm.add_patch(Rectangle((t * ps, b * mpb), ps, mpb, fill=False, edgecolor=col(k), lw=2.4 if on[k] else 2.0,
                                    ls=(0, (2.5, 1.5)) if on[k] else "-", zorder=4))
            items.append((k, f"{k} {m['tokens'][k].strip() or '_'}", b, t, ps, mpb))
        for k, text, x, y, va in _place_labels(items, dur, axis_in):
            axm.text(x, y, text, color=col(k), fontsize=10, ha="left", va=va, fontweight="bold", zorder=5, clip_on=True,
                     bbox=dict(boxstyle="square,pad=0.08", fc=BG, ec="none", alpha=0.75))
        _tidy(axm, dur, bottom=True)
        if i < len(prepared) - 1:
            axm.set_xlabel("")
        if window is not None:
            axm.axvspan(window[0], window[1], color=DEG, alpha=0.10, lw=0, zorder=1)
            axm.axvline(window[0], color=DEG, lw=1.2, ls="--", zorder=5); axm.axvline(window[1], color=DEG, lw=1.2, ls="--", zorder=5)
        w = f"  ·  WER {m['wer']:.0%}" if m.get("wer") is not None and np.isfinite(m["wer"]) else ""
        _label(axm, f"{m['name']}{w}  ·  {verdict(on, frac)}")
        _sentence_strip(axt, m["tokens"], on, axis_in)
    title = "\n".join(textwrap.fill(line, 95) for line in str(title).split("\n")) if title else ""
    if title:
        fig.suptitle(title, x=0.12, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.985, va="top")
    fig.text(0.12, 0.006, "Orange = loud, pale = quiet (sound energy, not damage). Blue bands = degraded audio (score above the threshold).\n"
             "Dashed box and underlined word = that word's evidence sits on degraded audio. Same colour and number = same word.",
             fontsize=10, color=TXT2, va="bottom")
    if out is not None:
        out = Path(out)
        fig.savefig(out, dpi=dpi)
        if out.suffix.lower() == ".png":
            fig.savefig(out.with_suffix(".svg"))
    return fig


def _mel(ax, L, dur):
    ax.imshow(L, origin="lower", aspect="auto", cmap=MEL_CMAP, extent=[0, L.shape[1] * 0.01, 0, 80], interpolation="nearest",
              vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
    ax.set_xlim(0, dur); ax.set_ylim(0, 80); ax.set_yticks([0, 40, 80]); ax.set_yticklabels(["0", "1.9 k", "8 k"]); ax.set_ylabel("Hz (mel)")


def from_attribution(att, name=None, wer=None):
    """Turn a ways.attribution.Attribution into the `models` entry ways_figure expects."""
    m = np.clip(att.delta, 0, None).astype(np.float64)
    top = []
    for k in range(m.shape[0]):
        if m[k].sum() <= 0:
            top.append(None)
        else:
            top.append(tuple(int(v) for v in np.unravel_index(int(m[k].argmax()), m[k].shape)))
    return dict(name=name or att.model, tokens=list(att.pieces), top=top, patch_s=att.patch_s, n_bands=att.n_bands, wer=wer)


# ----------------------------------------------------------------------------- the latent-space figure
def latent_figure(wav, sr=16000, quality=None, quality_hop=0.02, quality_classes=None, threshold=0.5,
                  cam=None, shift=None, degraded=None, loudness_db=None, summary=None, layer_hop=0.02,
                  model_name="", title="", out=None, dpi=DPI, width_in=12.0):
    """Same lanes as ways_figure on top (damage rows, spectrogram with degraded bands), then two heat maps
    with one row per encoder layer: Grad-CAM over time, and the relative shift of each layer's
    representation when the degraded stretch is erased (spill-over), and a per-layer summary strip.
    cam and shift are [layers, frames] arrays at layer_hop seconds per frame; degraded is a bool [frames]."""
    import textwrap

    wav = np.asarray(wav, np.float32); dur = len(wav) / sr
    L = logmel(wav, sr)
    spans = degraded_spans(quality, quality_hop, threshold)
    n_dmg_rows = 1
    if quality is not None:
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        n_dmg_rows = max(1, sum(int(q[:, j].max() >= threshold) for j in range(q.shape[1])) + (1 if q.shape[1] > 1 else 0))
    layers = cam.shape[0] if cam is not None else (shift.shape[0] if shift is not None else 1)
    hs = [0.22 * n_dmg_rows + 0.15, 2.2, 0.16 * layers + 0.3, 0.16 * layers + 0.3, 1.5]
    fig = plt.figure(figsize=(width_in, min(1.6 + sum(hs) * 0.62, 15)))
    gs = fig.add_gridspec(5, 1, height_ratios=hs, hspace=0.42, left=0.12, right=0.97, top=0.9, bottom=0.07)
    axq, axm, axc, axs, axb = [fig.add_subplot(gs[i]) for i in range(5)]
    if quality is not None:
        _damage_lane(axq, quality, quality_hop, quality_classes, threshold, dur, spans)
        _label(axq, f"Damage per {quality_hop * 1000:.0f} ms frame (dark blue = sure); boxed stretches reach {threshold:g} and count as degraded")
    else:
        axq.set_yticks([]); _label(axq, "Damage: none measured")
    _tidy(axq, dur)
    _mel(axm, L, dur); _bands(axm, spans); _tidy(axm, dur)
    if loudness_db is not None:
        ax2 = axm.twinx(); t = np.arange(len(loudness_db)) * layer_hop
        ax2.plot(t, loudness_db, color=TXT, lw=0.9, alpha=0.8); ax2.set_ylabel("loudness (dB)", fontsize=10); ax2.tick_params(labelsize=9, colors=TXT2)
        for s in ax2.spines.values():
            s.set_color(BORDER)
    _label(axm, f"{model_name}: spectrogram (orange = loud), degraded bands in blue, loudness per frame as the black line")
    for ax, M, name in ((axc, cam, "Grad-CAM per encoder layer (row 0 = input embedding, last row = top layer), each row scaled to its own maximum"),
                        (axs, shift, "Spill-over: how much each layer's representation moves when the degraded stretch is erased (relative)")):
        if M is None:
            ax.set_yticks([]); _label(ax, name + ": not computed"); _tidy(ax, dur); continue
        Mn = M / np.maximum(M.max(1, keepdims=True), 1e-12) if ax is axc else M
        ax.imshow(Mn, aspect="auto", cmap="Greys" if ax is axc else BLUE, extent=[0, M.shape[1] * layer_hop, M.shape[0], 0], interpolation="nearest",
                  vmin=0, vmax=1 if ax is axc else max(float(np.percentile(M, 98)), 1e-6))
        ax.set_yticks(np.arange(M.shape[0]) + 0.5); ax.set_yticklabels([str(i) for i in range(M.shape[0])], fontsize=9); ax.set_ylabel("layer", fontsize=10)
        for a, b in spans:
            ax.add_patch(Rectangle((a, 0), b - a, M.shape[0], fill=False, edgecolor=DEG, lw=1.2, zorder=3))
        _tidy(ax, dur, bottom=(ax is axs)); _label(ax, name)
    # per-layer summary: spill-over outside vs inside, CAM focus
    if summary:
        ls = [r["layer"] for r in summary]; w = 0.38
        axb.bar(np.array(ls) - w / 2, [r["shift_out"] for r in summary], width=w, color=DEG, alpha=0.85, label="spill-over outside the degraded stretch")
        axb.bar(np.array(ls) + w / 2, [r["shift_in"] for r in summary], width=w, color=SEQ[2], label="shift inside it")
        axb.set_xticks(ls); axb.set_xlabel("encoder layer"); axb.set_ylabel("relative shift")
        ax3 = axb.twinx(); ax3.plot(ls, [r["cam_focus"] for r in summary], color=CAT[1], marker="o", lw=1.6, label="CAM mass on degraded / share")
        ax3.axhline(1, color=CAT[1], lw=0.8, ls="--"); ax3.set_ylabel("CAM focus", color=CAT[1], fontsize=10); ax3.tick_params(labelsize=9, colors=CAT[1])
        h1, l1 = axb.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
        axb.legend(h1 + h2, l1 + l2, frameon=False, fontsize=10, loc="upper left")
        for s in list(axb.spines.values()) + list(ax3.spines.values()):
            s.set_color(BORDER)
        _label(axb, "Per layer: spill-over grows with depth when noise leaks into the representation of clean speech")
    title = "\n".join(textwrap.fill(line, 95) for line in str(title).split("\n")) if title else ""
    if title:
        fig.suptitle(title, x=0.12, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.985, va="top")
    fig.text(0.12, 0.006, "Spill-over is an erasure test (the same operator as the attribution). Grad-CAM is a gradient saliency, shown per layer as in AADD's\n"
             "all-layers reading, and should be read as a hint, not a measurement.", fontsize=10, color=TXT2, va="bottom")
    if out is not None:
        out = Path(out); fig.savefig(out, dpi=dpi)
        if out.suffix.lower() == ".png":
            fig.savefig(out.with_suffix(".svg"))
    return fig
