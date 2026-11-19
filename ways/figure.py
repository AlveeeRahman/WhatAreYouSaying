"""The WAYS figures as software: text and style are data, layout is measured, output is repeatable.

This file is self-contained (numpy and matplotlib only), so it can be copied into any Python project.

Two figures share the same lanes on one time axis:

  ways_figure    damage per frame (one row per kind, dark blue = sure), the spectrogram with the
                 degraded spans painted on it and one box per word, the transcript in the box
                 colours with words on degraded audio underlined, and in each model's row title the
                 count of words on degraded audio against chance.
  latent_figure  the same two top lanes, then Grad-CAM per encoder layer, the spill-over of each
                 layer under erasure of the degraded stretch, and a per-layer summary.

What makes them software rather than drawings:

  * Every heading, lane title, axis name, verdict sentence and caption is a field of FigureText or
    LatentText. Pass your own instance, or a dict of the fields you want to change, as `text=`.
    Fields with braces are str.format templates; the field names say what they receive.
  * Colours, fonts, sizes and the figure width live in FigureStyle, passed as `style=`.
  * Word labels and transcript lines are placed from measured text widths (matplotlib's TextPath),
    not character counts.
  * After drawing, the figure is checked for overlapping text (every Text artist, in pixels). If two
    texts overlap, the row spacing is widened and the check runs again; if that does not clear it,
    a LayoutError names the two texts and nothing is written to disk. `strict=False` turns the
    error into a warning and records the pairs on `fig.ways_overlaps`.
  * The same inputs give the same bytes: the font is the one bundled with matplotlib, the SVG hash
    salt is fixed, and the PNG and SVG metadata that would carry a date or a version are removed.

Minimal use, with your own detector and your own attribution method:

    from ways.figure import ways_figure
    ways_figure(wav, 16000,
                quality=probs, quality_hop=0.02, quality_classes=["noise", "reverb"],
                models=[{"name": "my-asr", "tokens": ["hello", "world"], "top": [(2, 14), (3, 20)],
                         "patch_s": 0.1, "n_bands": 8, "wer": 0.1}],
                text={"title": "my clip", "caption": "my own caption"}, out="my_clip.png")

`top` is, per token, (band index, time index) of the patch it used most, or None; bands are equal
groups of 80 mel bins from low to high, time indices are patches of `patch_s` seconds. `quality` is
any [n_frames, n_classes] array of probabilities at `quality_hop` seconds per frame; a single
column works too.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.textpath import TextPath

# ----------------------------------------------------------------------------- style and text
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE = LinearSegmentedColormap.from_list("ways_seq", SEQ)


@dataclass
class FigureStyle:
    """Colours, fonts and sizes. The defaults are the palette used in the README."""
    width_in: float = 12.0
    dpi: int = 300
    font: str = "DejaVu Sans"           # bundled with matplotlib, so the same on every machine
    font_size: float = 12.0
    label_size: float = 10.0            # word labels on the spectrogram
    strip_size: float = 12.0            # transcript strip
    background: str = "#fcfcfb"
    text: str = "#0b0b0b"
    text_soft: str = "#52514e"
    muted: str = "#77766f"
    border: str = "#e4e3df"
    degraded: str = "#0d366b"           # bands, rules, underlines
    band_alpha: float = 0.16
    mel_cmap: str = "YlOrBr"
    token_colours: tuple = ("#2a78d6", "#c2185b", "#1b9e77", "#7e3fc2")   # colour-blind safe adjacent pairs
    svg_hashsalt: str = "ways"


@dataclass
class FigureText:
    """Every string in ways_figure. Fields with braces are str.format templates."""
    title: str = ""
    damage_lane: str = "Damage per {hop_ms:.0f} ms frame (dark blue = sure). A boxed stretch reaches {threshold:g} and counts as degraded."
    damage_none: str = "no degradation scores given"
    damage_missing_lane: str = "Damage: none measured"
    damage_clean: str = "nothing reaches {threshold:g}: clean"
    damage_max_row: str = "max"
    spectrogram_only: str = "Spectrogram (orange = loud, pale = quiet). Blue bands = degraded audio."
    model_row: str = "{name}{wer}  ·  {verdict}"            # wer is "" or the rendered `wer` field
    wer: str = "  ·  WER {wer:.0%}"
    verdict_none: str = "no degraded audio in this clip ({K} words)"
    verdict_all: str = "every frame degraded, so all {K} words sit on degraded audio"
    verdict_mixed: str = "{k} of {K} words on degraded audio, {expected:.0f} expected by chance ({frac:.0%} of the clip is degraded)"
    verdict_no_quality: str = "{K} words"
    verdict_window: str = "{k} of {K} words on the known damaged audio, {expected:.0f} expected by chance ({frac:.0%} of the clip)"
    y_mel: str = "Hz (mel)"
    x_time: str = "time (s)"
    mel_ticks: tuple = ("0", "1.9 k", "8 k")
    caption: str = ("Orange = loud, pale = quiet (sound energy, not damage). Blue bands = degraded audio (score above the threshold). "
                    "Dashed box and underlined word = that word's evidence sits on degraded audio. Same colour and number = same word.")


@dataclass
class LatentText:
    """Every string in latent_figure."""
    title: str = ""
    damage_lane: str = FigureText.damage_lane
    damage_missing_lane: str = "Damage: none measured"
    damage_clean: str = FigureText.damage_clean
    damage_max_row: str = "max"
    spectrogram: str = "{name}: spectrogram (orange = loud), degraded bands in blue, loudness per frame as the thin black line"
    cam_lane: str = "Grad-CAM per encoder layer (row 0 = input embedding, last row = top layer), each row scaled to its own maximum"
    cam_missing: str = "Grad-CAM: not computed"
    shift_lane: str = "Spill-over: how much each layer's representation moves when the degraded stretch is erased (relative)"
    shift_missing: str = "Spill-over: not computed"
    summary_lane: str = "Per layer: spill-over outside the degraded stretch grows with depth when noise leaks into clean speech"
    legend_out: str = "spill-over outside the degraded stretch"
    legend_in: str = "shift inside it"
    legend_cam: str = "CAM mass on degraded / share"
    y_mel: str = "Hz (mel)"
    y_loud: str = "loudness (dB)"
    y_layer: str = "layer"
    y_shift: str = "relative shift"
    y_focus: str = "CAM focus"
    x_time: str = "time (s)"
    x_layer: str = "encoder layer"
    mel_ticks: tuple = ("0", "1.9 k", "8 k")
    caption: str = ("Spill-over is an erasure test (the same operator as the attribution). Grad-CAM is a gradient saliency, shown per layer "
                    "as in AADD's all-layers reading, and should be read as a hint, not a measurement.")
    token_lane: str = ("Gradient saliency of each token at the top encoder layer (grey, row scaled to its own maximum). "
                       "Coloured tick = the patch the erasure test ranked first for that token.")
    token_layers_lane: str = "Share of each token's saliency that falls on the degraded stretch, layer by layer (dark blue = all of it)"
    y_token: str = "token"
    caption_tokens: str = ("One row per token of the model's own transcript. Grey = gradient times activation per frame (Grad-CAM without "
                           "the time pooling, which loses the where). Where the grey mass and the coloured tick agree, the two methods name "
                           "the same audio. A row smeared across the clip is a token the model wrote from context, not from any one sound.")


def _merge(cls, given):
    """A dataclass instance from the default, an instance, or a dict of overrides."""
    if given is None:
        return cls()
    if isinstance(given, cls):
        return given
    if isinstance(given, dict):
        unknown = set(given) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
        return replace(cls(), **given)
    raise TypeError(f"{cls.__name__} or dict expected, got {type(given).__name__}")


class LayoutError(RuntimeError):
    """Two pieces of text overlap and widening the rows did not separate them."""


def _rc(style: FigureStyle):
    rcParams.update({"font.family": style.font, "font.size": style.font_size, "axes.titlesize": style.font_size,
                     "axes.labelsize": style.font_size, "xtick.labelsize": style.font_size, "ytick.labelsize": style.font_size,
                     "legend.fontsize": style.font_size, "figure.facecolor": style.background, "axes.facecolor": style.background,
                     "axes.edgecolor": style.border, "xtick.color": style.text_soft, "ytick.color": style.text_soft,
                     "axes.labelcolor": style.text_soft, "svg.fonttype": "none", "svg.hashsalt": style.svg_hashsalt,
                     "path.simplify": False})


# ----------------------------------------------------------------------------- measuring text
def text_width_pt(s: str, size: float, weight: str = "normal", font: str = "DejaVu Sans") -> float:
    """Width of a string in points at a font size, from the glyph outlines (no renderer needed)."""
    if not s:
        return 0.0
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=font, weight=weight)).get_extents().width


def wrap_to_width(s: str, width_pt: float, size: float, weight: str = "normal", font: str = "DejaVu Sans") -> str:
    """Wrap on spaces so that no line is wider than width_pt. Explicit line breaks are kept."""
    if "\n" in s:
        return "\n".join(wrap_to_width(part, width_pt, size, weight, font) for part in s.split("\n"))
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and text_width_pt(cand, size, weight, font) > width_pt:
            lines.append(cur); cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def text_overlaps(fig, min_px: float = 0.5) -> list[tuple[str, str]]:
    """Pairs of visible, non-empty text artists whose pixel boxes overlap (after a draw). A text that runs
    past the page edge is reported as a pair with "page edge"."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, Hpx = fig.bbox.width, fig.bbox.height
    hidden = set()                                   # tick labels outside their axis range are never drawn
    for ax in fig.get_axes():
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo <= tick.get_loc() <= hi):
                    hidden.update((id(tick.label1), id(tick.label2)))
    items = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in hidden:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((t.get_text().replace("\n", " ")[:60], bb))
    out = []
    for name, bb in items:
        if bb.x0 < -min_px or bb.y0 < -min_px or bb.x1 > W + min_px or bb.y1 > Hpx + min_px:
            out.append((name, "page edge"))
    for i in range(len(items)):
        ai, a = items[i]
        for j in range(i + 1, len(items)):
            bj, b = items[j]
            if a.x0 < b.x1 - min_px and b.x0 < a.x1 - min_px and a.y0 < b.y1 - min_px and b.y0 < a.y1 - min_px:
                out.append((ai, bj))
    return out


def _settle(fig, gs, strict: bool, rounds: int = 8, step: float = 0.12):
    """Widen row spacing until no two texts overlap, or fail loudly."""
    hspace = gs.get_subplot_params(fig).hspace or 0.3
    for _ in range(rounds):
        ov = text_overlaps(fig)
        if not ov:
            fig.ways_overlaps = []
            return
        if all(b == "page edge" for _, b in ov):     # wider rows cannot pull a text back onto the page
            break
        hspace += step
        gs.update(hspace=hspace)
    ov = text_overlaps(fig)
    fig.ways_overlaps = ov
    if ov and strict:
        raise LayoutError("overlapping text after widening the rows: " + "; ".join(f"{a!r} / {b!r}" for a, b in ov[:6]))
    if ov:
        warnings.warn("figure has overlapping text: " + "; ".join(f"{a!r} / {b!r}" for a, b in ov[:6]), stacklevel=2)


def _save(fig, out, dpi):
    """Write PNG (and an SVG twin) without the metadata that changes between runs."""
    out = Path(out)
    if out.suffix.lower() == ".png":
        fig.savefig(out, dpi=dpi, metadata={"Software": None})
        fig.savefig(out.with_suffix(".svg"), metadata={"Date": None, "Creator": None})
    elif out.suffix.lower() == ".svg":
        fig.savefig(out, metadata={"Date": None, "Creator": None})
    else:
        fig.savefig(out, dpi=dpi)


# ----------------------------------------------------------------------------- audio -> log-mel (numpy only)
def _slaney_mel_edges(n_mels, fmin, fmax):
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
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = _slaney_mel_edges(n_mels, 0.0, fmax or sr / 2)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.maximum(0, np.minimum((freqs - lo) / max(c - lo, 1e-9), (hi - freqs) / max(hi - c, 1e-9))) * 2.0 / (hi - lo)
    L = np.log10(np.maximum(fb @ power.T, 1e-10))[:, : len(wav) // hop]
    return np.maximum(L, L.max() - 8)


# ----------------------------------------------------------------------------- degraded spans
def degraded_spans(quality, hop, threshold=0.5, min_gap=0.1):
    """(start_s, end_s) stretches where any class is at or above threshold; gaps under min_gap merged."""
    if quality is None:
        return []
    q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
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
    if quality is None:
        return None
    q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
    fl = (q >= threshold).any(1).astype(np.float32)
    per = max(1, int(round(patch_s / hop)))
    f = np.zeros(n_t, np.float32)
    for t in range(n_t):
        seg = fl[t * per:(t + 1) * per]
        f[t] = seg.mean() if len(seg) else 0.0
    return f


def verdict(on_degraded, frac, text: FigureText | None = None):
    """The row-title sentence: k of K words on degraded audio against chance."""
    T = text or FigureText()
    K = int(sum(v is not None for v in on_degraded)); k = int(sum(bool(v) for v in on_degraded))
    if frac is None:
        return T.verdict_no_quality.format(K=K)
    if frac <= 0:
        return T.verdict_none.format(K=K)
    if frac >= 1:
        return T.verdict_all.format(K=K)
    return T.verdict_mixed.format(k=k, K=K, expected=frac * K, frac=frac)


# ----------------------------------------------------------------------------- drawing helpers
class _Canvas:
    """Shared state for one figure: style, geometry and the measured-text helpers."""

    def __init__(self, style, dur, width_in, right=0.97):
        self.S, self.dur = style, dur
        self.left, self.right = 0.12, right
        self.axis_in = width_in * (self.right - self.left)
        self.axis_pt = self.axis_in * 72
        self.mel_axis_in = 2.0          # set once the grid is known; used to turn points into mel units

    def col(self, k):
        return self.S.token_colours[k % len(self.S.token_colours)]

    def tidy(self, ax, bottom=False, x_label=""):
        ax.set_xlim(0, self.dur)
        for s in ax.spines.values():
            s.set_color(self.S.border)
        ax.tick_params(labelbottom=bottom, colors=self.S.text_soft)
        if bottom:
            ax.set_xlabel(x_label, labelpad=2)

    def label(self, ax, s):
        ax.set_title(wrap_to_width(s, self.axis_pt, self.S.font_size, font=self.S.font), loc="left", fontsize=self.S.font_size,
                     color=self.S.text_soft, pad=4)

    def mel(self, ax, L, y_label, ticks):
        ax.imshow(L, origin="lower", aspect="auto", cmap=self.S.mel_cmap, extent=[0, L.shape[1] * 0.01, 0, 80], interpolation="nearest",
                  vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
        ax.set_xlim(0, self.dur); ax.set_ylim(0, 80); ax.set_yticks([0, 40, 80]); ax.set_yticklabels(list(ticks)); ax.set_ylabel(y_label)
        ax.get_yticklabels()[0].set_va("bottom")

    def bands(self, ax, spans):
        for a, b in spans:
            ax.axvspan(a, b, color=self.S.degraded, alpha=self.S.band_alpha, lw=0, zorder=2)
            y0, y1 = ax.get_ylim()
            ax.add_patch(Rectangle((a, y1 - (y1 - y0) * 0.035), b - a, (y1 - y0) * 0.035, facecolor=self.S.degraded, edgecolor="none", zorder=3))

    def damage_lane(self, ax, quality, hop, classes, threshold, spans, clean_text, max_row):
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        classes = list(classes) if classes is not None else ([f"class {i}" for i in range(q.shape[1])] if q.shape[1] > 1 else ["degradation"])
        active = [j for j in range(q.shape[1]) if q[:, j].max() >= threshold]
        rows = [(classes[j], q[:, j]) for j in active]
        if q.shape[1] > 1:
            rows.append((max_row, q.max(1)))
        elif not rows:
            rows = [(classes[0], q[:, 0])]
        H = np.stack([r for _, r in rows])
        ax.imshow(np.clip(H, 0, 1), aspect="auto", cmap=BLUE, extent=[0, H.shape[1] * hop, len(rows), 0], interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([n.replace("_", " ") for n, _ in rows], fontsize=self.S.label_size)
        for a, b in spans:
            ax.add_patch(Rectangle((a, 0), b - a, len(rows), fill=False, edgecolor=self.S.degraded, lw=1.2, zorder=3))
        if not active:
            ax.text(self.dur / 2, len(rows) / 2, clean_text, ha="center", va="center", color=self.S.muted, fontsize=self.S.label_size + 1)

    def place_labels(self, items, mel_h=80):
        """Non-overlapping positions beside the boxes, from measured text widths. items: [(k, text, b, t, patch_s, mpb)]."""
        line_h = (self.S.label_size * 1.3 / 72) / max(self.mel_axis_in, 0.5) * mel_h      # one text line, in mel units
        pt_to_data = self.dur / self.axis_pt
        placed, out = [], []
        for k, text, b, t, ps, mpb in sorted(items, key=lambda r: (r[3], r[0])):
            w = (text_width_pt(text, self.S.label_size, "bold", self.S.font) + 4) * pt_to_data
            x0 = min(max(t * ps + ps / 2 - w / 2, 0.0), max(self.dur - w, 0.0))
            cands = []
            for step in range(6):
                cands.append((x0, (b + 1) * mpb + 0.5 + step * line_h, "bottom"))
                cands.append((x0, b * mpb - 0.5 - step * line_h, "top"))
            for dx in (0.6, -0.6, 1.2, -1.2):
                for step in range(3):
                    xx = min(max(x0 + dx * w, 0.0), max(self.dur - w, 0.0))
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

    def sentence_lines(self, tokens):
        """Transcript laid out in reading order, wrapped by measured width (inches)."""
        lines, x, cur = [], 0.0, []
        for k, tok in enumerate(tokens):
            text = f"{k} {tok.strip() or '_'}"
            w = 0.16 + text_width_pt(text, self.S.strip_size, "bold", self.S.font) / 72 + 0.22
            if cur and x + w > self.axis_in:
                lines.append(cur); cur, x = [], 0.0
            cur.append((k, x, text, w)); x += w
        if cur:
            lines.append(cur)
        return lines

    def sentence_strip(self, ax, tokens, on_degraded):
        lines = self.sentence_lines(tokens)
        ax.set_xlim(0, self.axis_in); ax.set_ylim(-0.15, len(lines) + 0.15); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        for li, line in enumerate(lines):
            y = len(lines) - 1 - li + 0.5
            for k, x, text, w in line:
                ax.add_patch(Rectangle((x, y - 0.3), 0.16, 0.6, facecolor=self.col(k), edgecolor="white", lw=0.5))
                ax.text(x + 0.21, y, text, color=self.col(k), fontsize=self.S.strip_size, ha="left", va="center", fontweight="bold")
                if on_degraded is not None and on_degraded[k]:
                    ax.add_patch(Rectangle((x, y - 0.46), w - 0.2, 0.07, facecolor=self.S.degraded, edgecolor="none"))
        return len(lines)


def _title_and_caption(fig, C, title, caption):
    """Figure title (top left, wrapped) and caption (bottom left, wrapped); returns their heights in inches."""
    S = C.S
    tl = wrap_to_width(title, C.axis_pt, S.font_size + 1, "bold", S.font) if title else ""
    cl = wrap_to_width(caption, C.axis_pt, S.label_size, font=S.font) if caption else ""
    if tl:
        fig.suptitle(tl, x=C.left, ha="left", fontsize=S.font_size + 1, fontweight="bold", color=S.text, y=0.995, va="top")
    if cl:
        fig.text(C.left, 0.005, cl, fontsize=S.label_size, color=S.text_soft, va="bottom")
    t_in = (tl.count("\n") + 1) * (S.font_size + 1) * 1.3 / 72 if tl else 0.0
    c_in = (cl.count("\n") + 1) * S.label_size * 1.3 / 72 if cl else 0.0
    return t_in, c_in


# ----------------------------------------------------------------------------- the figure
def ways_figure(wav, sr=16000, quality=None, quality_hop=0.02, quality_classes=None, threshold=0.5, models=None, window=None,
                title=None, out=None, text=None, style=None, strict=True):
    """Draw the standard WAYS figure. Returns the matplotlib Figure; writes `out` (PNG plus an SVG twin) when given.

    models: list of dicts with name, tokens, top ((band, time) or None per token), patch_s (default 0.1),
    n_bands (default 8), wer (optional). window: (start_s, end_s) of a known damaged span, or a list of
    such spans, drawn dashed; when given, the words are counted against these spans, not the detector.
    text: FigureText or dict of overrides; `title` is a shortcut for text["title"]. style: FigureStyle or dict.
    """
    T, S = _merge(FigureText, text), _merge(FigureStyle, style)
    if title is not None:
        T = replace(T, title=title)
    _rc(S)
    wav = np.asarray(wav, np.float32); dur = len(wav) / sr
    C = _Canvas(S, dur, S.width_in)
    L = logmel(wav, sr)
    spans = degraded_spans(quality, quality_hop, threshold)
    frac = None
    if quality is not None:
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        n = int(np.ceil(dur / quality_hop)); fl = (q[:n] >= threshold).any(1)
        frac = float(fl.mean()) if len(fl) else 0.0
    windows = [] if window is None else ([tuple(window)] if np.isscalar(window[0]) else [tuple(w) for w in window])
    if windows:                                     # known damaged spans beat the detector as the reference
        frac = float(sum(min(b, dur) - max(a, 0.0) for a, b in windows) / dur)
    models = models or []
    prepared = []
    for m in models:
        ps = m.get("patch_s", 0.1); n_t = int(np.ceil(dur / ps))
        if windows:
            on = [None if tp is None else any(a <= (tp[1] + 0.5) * ps < b for a, b in windows) for tp in m["top"]]
        else:
            f = _flag_per_patch(quality, quality_hop, threshold, ps, n_t)
            on = [None if tp is None else (bool(f[min(tp[1], n_t - 1)] >= 0.5) if f is not None else False) for tp in m["top"]]
        prepared.append((m, on, len(C.sentence_lines(m["tokens"]))))
    n_dmg_rows = 1
    if quality is not None:
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        n_dmg_rows = max(1, sum(int(q[:, j].max() >= threshold) for j in range(q.shape[1])) + (1 if q.shape[1] > 1 else 0))
    # lane heights in inches: a damage row needs 1.6 text lines, a transcript line 1.9, the spectrogram 2.4
    row_in = S.label_size * 1.6 / 72; line_in = S.strip_size * 1.9 / 72
    hs = [row_in * n_dmg_rows + 0.08]
    for _, _, nl in prepared:
        hs += [2.4, line_in * nl + 0.12]
    if not prepared:
        hs += [2.4]
    gap_in = S.font_size * 2.6 / 72                        # room for a lane title between lanes
    t_in = (wrap_to_width(T.title, C.axis_pt, S.font_size + 1, "bold", S.font).count("\n") + 1) * (S.font_size + 1) * 1.3 / 72 if T.title else 0.0
    c_in = (wrap_to_width(T.caption, C.axis_pt, S.label_size, font=S.font).count("\n") + 1) * S.label_size * 1.3 / 72 if T.caption else 0.0
    H = min(t_in + 0.35 + sum(hs) + gap_in * len(hs) + c_in + 0.6, 40.0)
    fig = plt.figure(figsize=(S.width_in, H))
    _title_and_caption(fig, C, T.title, T.caption)
    top, bottom = 1 - (t_in + 0.35) / H, (c_in + 0.6) / H
    mean_in = sum(hs) / len(hs)
    gs = fig.add_gridspec(len(hs), 1, height_ratios=hs, hspace=gap_in / mean_in, left=C.left, right=C.right, top=top, bottom=bottom)
    C.mel_axis_in = 2.4
    axq = fig.add_subplot(gs[0])
    if quality is not None:
        C.damage_lane(axq, quality, quality_hop, quality_classes, threshold, spans, T.damage_clean.format(threshold=threshold), T.damage_max_row)
        C.label(axq, T.damage_lane.format(hop_ms=quality_hop * 1000, threshold=threshold))
    else:
        axq.text(dur / 2, 0.5, T.damage_none, ha="center", va="center", color=S.muted); axq.set_yticks([])
        C.label(axq, T.damage_missing_lane)
    C.tidy(axq)
    for a_, b_ in windows:
        axq.axvspan(a_, b_, color=S.degraded, alpha=0.10, lw=0, zorder=1)
    if not prepared:
        axm = fig.add_subplot(gs[1]); C.mel(axm, L, T.y_mel, T.mel_ticks); C.bands(axm, spans); C.tidy(axm, True, T.x_time)
        C.label(axm, T.spectrogram_only)
    for i, (m, on, _nl) in enumerate(prepared):
        ps, B = m.get("patch_s", 0.1), m.get("n_bands", 8); mpb = 80 // B
        axm = fig.add_subplot(gs[1 + 2 * i]); axt = fig.add_subplot(gs[2 + 2 * i])
        C.mel(axm, L, T.y_mel, T.mel_ticks); C.bands(axm, spans)
        items = []
        for k, tp in enumerate(m["top"]):
            if tp is None:
                continue
            b, t = tp
            axm.add_patch(Rectangle((t * ps, b * mpb), ps, mpb, fill=False, edgecolor=C.col(k), lw=2.4 if on[k] else 2.0,
                                    ls=(0, (2.5, 1.5)) if on[k] else "-", zorder=4))
            items.append((k, f"{k} {m['tokens'][k].strip() or '_'}", b, t, ps, mpb))
        for k, lab, x, y, va in C.place_labels(items):
            axm.text(x, y, lab, color=C.col(k), fontsize=S.label_size, ha="left", va=va, fontweight="bold", zorder=5, clip_on=True,
                     bbox=dict(boxstyle="square,pad=0.08", fc=S.background, ec="none", alpha=0.75))
        C.tidy(axm, True, T.x_time if i == len(prepared) - 1 else "")
        for a_, b_ in windows:
            axm.axvspan(a_, b_, color=S.degraded, alpha=0.10, lw=0, zorder=1)
            axm.axvline(a_, color=S.degraded, lw=1.2, ls="--", zorder=5); axm.axvline(b_, color=S.degraded, lw=1.2, ls="--", zorder=5)
        w = T.wer.format(wer=m["wer"]) if m.get("wer") is not None and np.isfinite(m["wer"]) else ""
        C.label(axm, T.model_row.format(name=m["name"], wer=w, verdict=verdict(on, frac, T if not windows else replace(T, verdict_mixed=T.verdict_window))))
        C.sentence_strip(axt, m["tokens"], on)
    _settle(fig, gs, strict)
    if out is not None:
        _save(fig, out, S.dpi)
    return fig


def from_attribution(att, name=None, wer=None):
    """Turn a ways.attribution.Attribution into the `models` entry ways_figure expects."""
    m = np.clip(att.delta, 0, None).astype(np.float64)
    top = []
    for k in range(m.shape[0]):
        top.append(None if m[k].sum() <= 0 else tuple(int(v) for v in np.unravel_index(int(m[k].argmax()), m[k].shape)))
    return dict(name=name or att.model, tokens=list(att.pieces), top=top, patch_s=att.patch_s, n_bands=att.n_bands, wer=wer)


# ----------------------------------------------------------------------------- the latent-space figure
def latent_figure(wav, sr=16000, quality=None, quality_hop=0.02, quality_classes=None, threshold=0.5, cam=None, shift=None,
                  degraded=None, loudness_db=None, summary=None, layer_hop=0.02, model_name="", title=None, out=None,
                  text=None, style=None, strict=True):
    """Damage rows and the spectrogram as in ways_figure, then Grad-CAM per encoder layer, the spill-over per
    layer under erasure of the degraded stretch, and a per-layer summary. cam and shift are [layers, frames]
    at layer_hop seconds per frame; summary is ways.latent's per-layer list. text: LatentText or dict."""
    T, S = _merge(LatentText, text), _merge(FigureStyle, style)
    if title is not None:
        T = replace(T, title=title)
    _rc(S)
    wav = np.asarray(wav, np.float32); dur = len(wav) / sr
    C = _Canvas(S, dur, S.width_in, right=0.93)
    L = logmel(wav, sr)
    spans = degraded_spans(quality, quality_hop, threshold)
    n_dmg_rows = 1
    if quality is not None:
        q = np.asarray(quality, np.float32); q = q[:, None] if q.ndim == 1 else q
        n_dmg_rows = max(1, sum(int(q[:, j].max() >= threshold) for j in range(q.shape[1])) + (1 if q.shape[1] > 1 else 0))
    layers = cam.shape[0] if cam is not None else (shift.shape[0] if shift is not None else 1)
    row_in = S.label_size * 1.6 / 72
    hs = [row_in * n_dmg_rows + 0.08, 2.2, row_in * layers + 0.1, row_in * layers + 0.1, 1.7]
    gap_in = S.font_size * 2.6 / 72
    t_in = (wrap_to_width(T.title, C.axis_pt, S.font_size + 1, "bold", S.font).count("\n") + 1) * (S.font_size + 1) * 1.3 / 72 if T.title else 0.0
    c_in = (wrap_to_width(T.caption, C.axis_pt, S.label_size, font=S.font).count("\n") + 1) * S.label_size * 1.3 / 72 if T.caption else 0.0
    H = min(t_in + 0.35 + sum(hs) + gap_in * 5 + 0.6 + c_in + 0.6, 40.0)     # extra 0.6 for the legend row above the summary
    fig = plt.figure(figsize=(S.width_in, H))
    _title_and_caption(fig, C, T.title, T.caption)
    mean_in = sum(hs) / 5
    gs = fig.add_gridspec(5, 1, height_ratios=hs, hspace=gap_in / mean_in, left=C.left, right=C.right, top=1 - (t_in + 0.35) / H, bottom=(c_in + 0.6) / H)
    axq, axm, axc, axs, axb = [fig.add_subplot(gs[i]) for i in range(5)]
    if quality is not None:
        C.damage_lane(axq, quality, quality_hop, quality_classes, threshold, spans, T.damage_clean.format(threshold=threshold), T.damage_max_row)
        C.label(axq, T.damage_lane.format(hop_ms=quality_hop * 1000, threshold=threshold))
    else:
        axq.set_yticks([]); C.label(axq, T.damage_missing_lane)
    C.tidy(axq)
    C.mel(axm, L, T.y_mel, T.mel_ticks); C.bands(axm, spans); C.tidy(axm)
    if loudness_db is not None:
        ax2 = axm.twinx(); t = np.arange(len(loudness_db)) * layer_hop
        ax2.plot(t, loudness_db, color=S.text, lw=0.7, alpha=0.55); ax2.set_ylabel(T.y_loud, fontsize=S.label_size)
        ax2.tick_params(labelsize=S.label_size - 1, colors=S.text_soft)
        for s in ax2.spines.values():
            s.set_color(S.border)
    C.label(axm, T.spectrogram.format(name=model_name))
    for ax, M, name, missing in ((axc, cam, T.cam_lane, T.cam_missing), (axs, shift, T.shift_lane, T.shift_missing)):
        if M is None:
            ax.set_yticks([]); C.label(ax, missing); C.tidy(ax); continue
        Mn = M / np.maximum(M.max(1, keepdims=True), 1e-12) if ax is axc else M
        ax.imshow(Mn, aspect="auto", cmap="Greys" if ax is axc else BLUE, extent=[0, M.shape[1] * layer_hop, M.shape[0], 0],
                  interpolation="nearest", vmin=0, vmax=1 if ax is axc else max(float(np.percentile(M, 98)), 1e-6))
        ax.set_yticks(np.arange(M.shape[0]) + 0.5); ax.set_yticklabels([str(i) for i in range(M.shape[0])], fontsize=S.label_size - 1)
        ax.set_ylabel(T.y_layer, fontsize=S.label_size)
        for a, b in spans:
            ax.add_patch(Rectangle((a, 0), b - a, M.shape[0], fill=False, edgecolor=S.degraded, lw=1.2, zorder=3))
        C.tidy(ax, bottom=(ax is axs), x_label=T.x_time); C.label(ax, name)
    if summary:
        ls = [r["layer"] for r in summary]; w = 0.38
        axb.bar(np.array(ls) - w / 2, [r["shift_out"] for r in summary], width=w, color=S.degraded, alpha=0.85, label=T.legend_out)
        axb.bar(np.array(ls) + w / 2, [r["shift_in"] for r in summary], width=w, color=SEQ[2], label=T.legend_in)
        axb.set_xticks(ls); axb.set_xlabel(T.x_layer); axb.set_ylabel(T.y_shift)
        axb.set_ylim(0, max(max(r["shift_in"] for r in summary), 1e-3) * 1.45)
        ax3 = axb.twinx(); ax3.plot(ls, [r["cam_focus"] for r in summary], color=S.token_colours[1], marker="o", lw=1.6, label=T.legend_cam)
        ax3.axhline(1, color=S.token_colours[1], lw=0.8, ls="--"); ax3.set_ylabel(T.y_focus, color=S.token_colours[1], fontsize=S.label_size)
        ax3.tick_params(labelsize=S.label_size - 1, colors=S.token_colours[1])
        h1, l1 = axb.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
        axb.legend(h1 + h2, l1 + l2, frameon=False, fontsize=S.label_size - 1, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        for s in list(axb.spines.values()) + list(ax3.spines.values()):
            s.set_color(S.border)
        axb.set_title(wrap_to_width(T.summary_lane, C.axis_pt, S.font_size, font=S.font), loc="left", fontsize=S.font_size, color=S.text_soft, pad=4)
    _settle(fig, gs, strict)
    if out is not None:
        _save(fig, out, S.dpi)
    return fig


# ----------------------------------------------------------------------------- granular aud-cam figures
# Each lane of latent_figure on its own, for when one view has to be read closely or placed alone.
# They share the text (LatentText), the style, the measured layout and the overlap check.


def _layer_lane(wav, sr, M, cmap, vmax, spans, lane_label, title, layer_hop, T, S, out, strict):
    """One [layers, frames] heat map with the degraded spans boxed, under a title, over the caption."""
    _rc(S)
    wav = np.asarray(wav, np.float32); dur = len(wav) / sr
    C = _Canvas(S, dur, S.width_in, right=0.93)
    lane_in = 0.16 * M.shape[0] + 0.5
    fig = plt.figure(figsize=(S.width_in, 1.0))                 # resized once the texts are measured
    t_in, c_in = _title_and_caption(fig, C, title, T.caption)
    H = t_in + 0.45 + lane_in + c_in + 0.7
    fig.set_size_inches(S.width_in, H)
    gs = fig.add_gridspec(1, 1, left=C.left, right=C.right, top=1 - (t_in + 0.45) / H, bottom=(c_in + 0.7) / H)
    ax = fig.add_subplot(gs[0])
    ax.imshow(M, aspect="auto", cmap=cmap, extent=[0, M.shape[1] * layer_hop, M.shape[0], 0], interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(M.shape[0]) + 0.5); ax.set_yticklabels([str(i) for i in range(M.shape[0])], fontsize=S.label_size - 1)
    ax.set_ylabel(T.y_layer, fontsize=S.label_size)
    for a, b in spans:
        ax.add_patch(Rectangle((a, 0), b - a, M.shape[0], fill=False, edgecolor=S.degraded, lw=1.2, zorder=3))
    C.tidy(ax, True, T.x_time); C.label(ax, lane_label)
    _settle(fig, gs, strict)
    if out is not None:
        _save(fig, out, S.dpi)
    return fig


def cam_figure(wav, sr, cam, quality=None, quality_hop=0.02, threshold=0.5, layer_hop=0.02, model_name="", title=None, out=None,
               text=None, style=None, strict=True):
    """Grad-CAM per encoder layer on its own, each row scaled to its own maximum, degraded spans boxed."""
    T, S = _merge(LatentText, text), _merge(FigureStyle, style)
    cam = np.asarray(cam, np.float32)
    M = cam / np.maximum(cam.max(1, keepdims=True), 1e-12)
    title = title if title is not None else (T.title or f"{model_name}: Grad-CAM per encoder layer".strip(": "))
    return _layer_lane(wav, sr, M, "Greys", 1.0, degraded_spans(quality, quality_hop, threshold), T.cam_lane, title, layer_hop, T, S, out, strict)


def spillover_figure(wav, sr, shift, quality=None, quality_hop=0.02, threshold=0.5, layer_hop=0.02, model_name="", title=None, out=None,
                     text=None, style=None, strict=True):
    """Spill-over per layer on its own (relative shift under erasure of the degraded stretch), degraded spans boxed."""
    T, S = _merge(LatentText, text), _merge(FigureStyle, style)
    shift = np.asarray(shift, np.float32)
    vmax = max(float(np.percentile(shift, 98)), 1e-6)
    title = title if title is not None else (T.title or f"{model_name}: spill-over under erasure of the degraded stretch".strip(": "))
    return _layer_lane(wav, sr, shift, BLUE, vmax, degraded_spans(quality, quality_hop, threshold), T.shift_lane, title, layer_hop, T, S, out, strict)


def layer_summary_figure(summary, model_name="", title=None, out=None, text=None, style=None, strict=True):
    """The per-layer numbers on their own: shift inside and outside the degraded stretch as paired bars (top),
    CAM focus as a line against 1 = no preference (bottom). One axis per panel, no shared y."""
    T, S = _merge(LatentText, text), _merge(FigureStyle, style)
    _rc(S)
    ls = np.array([r["layer"] for r in summary]); w = 0.38
    C = _Canvas(S, float(len(ls)), S.width_in, right=0.97)
    title = title if title is not None else (T.title or f"{model_name}: per-layer summary".strip(": "))
    fig = plt.figure(figsize=(S.width_in, 1.0))
    t_in, c_in = _title_and_caption(fig, C, title, T.caption)
    H = t_in + 0.45 + 2.2 + 0.6 + 1.6 + c_in + 0.7
    fig.set_size_inches(S.width_in, H)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], left=C.left, right=C.right, top=1 - (t_in + 0.45) / H, bottom=(c_in + 0.7) / H, hspace=0.6 / 1.9)
    axb, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    axb.bar(ls - w / 2, [r["shift_out"] for r in summary], width=w, color=S.degraded, alpha=0.85, label=T.legend_out)
    axb.bar(ls + w / 2, [r["shift_in"] for r in summary], width=w, color=SEQ[2], label=T.legend_in)
    axb.set_xticks(ls); axb.set_ylabel(T.y_shift, fontsize=S.label_size)
    axb.set_ylim(0, max(max(r["shift_in"] for r in summary), max(r["shift_out"] for r in summary), 1e-3) * 1.35)
    axb.legend(frameon=False, fontsize=S.label_size - 1, loc="upper left", ncol=2)
    axc.plot(ls, [r["cam_focus"] for r in summary], color=S.token_colours[1], marker="o", lw=1.6)
    axc.axhline(1, color=S.muted, lw=0.8, ls="--")
    axc.set_xticks(ls); axc.set_xlabel(T.x_layer, labelpad=2); axc.set_ylabel(T.y_focus, fontsize=S.label_size)
    for ax in (axb, axc):
        for sp in ax.spines.values():
            sp.set_color(S.border)
        ax.tick_params(colors=S.text_soft, labelsize=S.label_size - 1)
    axb.tick_params(labelbottom=False)
    C.label(axb, T.summary_lane); C.label(axc, T.legend_cam)
    _settle(fig, gs, strict)
    if out is not None:
        _save(fig, out, S.dpi)
    return fig


def token_cam_figure(wav, sr, tcam, pieces, degraded, layer=-1, layer_hop=0.02, erase_t=None, window=None, model_name="", title=None,
                     out=None, text=None, style=None, strict=True):
    """Per-token saliency: the spectrogram, one lane per token at `layer` (default the top layer) with the erasure
    test's first-ranked patch as a coloured tick, and the share of each token's map on the degraded frames per layer.
    tcam is ways.latent.token_cam's [tokens, layers, frames]; degraded is the [frames] bool of erased frames;
    erase_t an optional time in seconds per token; window an optional (start, end) or list of them, drawn dashed."""
    T, S = _merge(LatentText, text), _merge(FigureStyle, style)
    _rc(S)
    wav = np.asarray(wav, np.float32); dur = len(wav) / sr
    tcam = np.asarray(tcam, np.float32); K, L, n = tcam.shape
    d = np.asarray(degraded, bool)[:n]
    spans = degraded_spans(d.astype(np.float32), layer_hop, 0.5)
    wins = [] if window is None else ([tuple(window)] if np.ndim(window) == 1 else [tuple(w) for w in window])
    C = _Canvas(S, dur, S.width_in, right=0.93)
    Lm = logmel(wav, sr)
    title = title if title is not None else (T.title or f"{model_name}: gradient saliency per token".strip(": "))
    row_in = S.label_size * 1.7 / 72
    hs = [1.7, row_in * K + 0.3, row_in * K + 0.3]
    gap_in = S.font_size * 2.6 / 72
    fig = plt.figure(figsize=(S.width_in, 1.0))
    t_in, c_in = _title_and_caption(fig, C, title, T.caption_tokens)
    H = t_in + 0.45 + sum(hs) + gap_in * 3 + c_in + 0.7
    fig.set_size_inches(S.width_in, H)
    gs = fig.add_gridspec(3, 1, height_ratios=hs, hspace=gap_in / (sum(hs) / 3), left=C.left, right=C.right,
                          top=1 - (t_in + 0.45) / H, bottom=(c_in + 0.7) / H)
    axm, axt, axl = [fig.add_subplot(gs[i]) for i in range(3)]
    C.mel(axm, Lm, T.y_mel, T.mel_ticks); C.bands(axm, spans); C.tidy(axm); C.label(axm, T.spectrogram.format(name=model_name).split(",")[0])
    labels = [f"{k} {str(pc).strip() or '_'}" for k, pc in enumerate(pieces)]
    M = tcam[:, layer, :]; M = M / np.maximum(M.max(1, keepdims=True), 1e-12)
    axt.imshow(M, aspect="auto", cmap="Greys", extent=[0, n * layer_hop, K, 0], interpolation="nearest", vmin=0, vmax=1)
    if erase_t is not None:
        for k, x in enumerate(erase_t):
            if x is not None and np.isfinite(x):
                axt.plot([x, x], [k + 0.12, k + 0.88], color=C.col(k), lw=2.2, solid_capstyle="butt", zorder=4)
    for a, b in spans:
        axt.add_patch(Rectangle((a, 0), b - a, K, fill=False, edgecolor=S.degraded, lw=1.2, zorder=3))
    for ax in (axm, axt):
        for a, b in wins:
            ax.axvline(a, color=S.degraded, lw=1.0, ls=(0, (4, 3)), zorder=5); ax.axvline(b, color=S.degraded, lw=1.0, ls=(0, (4, 3)), zorder=5)
    axt.set_yticks(np.arange(K) + 0.5); axt.set_yticklabels(labels, fontsize=S.label_size - 1, fontweight="bold")
    for k, lab in enumerate(axt.get_yticklabels()):
        lab.set_color(C.col(k))
    axt.set_ylabel(T.y_token, fontsize=S.label_size)
    C.tidy(axt, True, T.x_time); C.label(axt, T.token_lane)
    share = np.array([[tcam[k, l][d].sum() if tcam[k, l].sum() > 0 else np.nan for l in range(L)] for k in range(K)])
    axl.imshow(np.nan_to_num(share), aspect="auto", cmap=BLUE, extent=[-0.5, L - 0.5, K, 0], interpolation="nearest", vmin=0, vmax=1)
    axl.axvline(-0.5, color=S.border, lw=0)
    axl.set_xticks(np.arange(L)); axl.set_xticklabels([str(l) for l in range(L)], fontsize=S.label_size - 1)
    axl.set_yticks(np.arange(K) + 0.5); axl.set_yticklabels(labels, fontsize=S.label_size - 1, fontweight="bold")
    for k, lab in enumerate(axl.get_yticklabels()):
        lab.set_color(C.col(k))
    axl.set_ylabel(T.y_token, fontsize=S.label_size); axl.set_xlabel(T.x_layer, labelpad=2)
    for sp in axl.spines.values():
        sp.set_color(S.border)
    axl.tick_params(colors=S.text_soft)
    C.label(axl, T.token_layers_lane)
    _settle(fig, gs, strict)
    if out is not None:
        _save(fig, out, S.dpi)
    return fig
