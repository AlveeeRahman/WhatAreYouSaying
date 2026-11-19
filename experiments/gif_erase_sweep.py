"""A GIF of the method on real data, in the order the code works (ways.erasure.PatchGrid.patches):
one patch is erased at a time, left to right along the lowest pitch band, then along the next band
up, until every patch of the grid has been tried once. Each patch is erased from the original
audio, so the previous patch is back when the next one goes. For each patch the bars show how much
less sure whisper-base became of every word, and the map of the word that reacts most fills in
underneath. Drawn in the hand-drawn figure style (cream ground, Kalam and Patrick Hand).
Writes docs/images/erase_sweep.gif. Usage: python experiments/gif_erase_sweep.py [data folder]"""
import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import AUDIO_DIR, data_root
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from ways.attribution import Attribution
from ways.erasure import PatchGrid
from ways.plots import logmel

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "docs/images"
for f in (IMG / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
PAGE, CARD, INK, SOFT, ORANGE, BLUE, PURPLE, GREEN, RED = "#FBF4E4", "#FFFCF4", "#2B2620", "#6B6158", "#D9780F", "#2F5FD0", "#7A48B5", "#2E8A4E", "#C8372D"
plt.rcParams.update({"font.family": "Patrick Hand", "font.size": 15, "figure.facecolor": PAGE, "axes.facecolor": CARD,
                     "axes.edgecolor": "#CDBFA5", "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT})
TITLE = dict(family="Kalam", weight="bold")

CLIP = "id10288_FIxC_PNJNkc_00003"
AUDIO = data_root(sys.argv[1] if len(sys.argv) > 1 else None) / AUDIO_DIR
att = Attribution.load(HERE / "results/example/whisper-base.npz")
wav, _ = sf.read(AUDIO / f"{CLIP}.wav", dtype="float32")
L = logmel(wav)
dur = att.duration; ps, B = att.patch_s, att.n_bands; mpb = 80 // B
K_SHOWN = 13
FRAME_MS, HOLD_FRAMES = 60, 30
k = K_SHOWN
words = [p.strip() or "_" for p in att.pieces]
K = len(words)
grid = PatchGrid(len(wav), ps, B)
order = grid.patches()
assert len(order) == B * att.n_t == att.delta[k].size
pos = np.clip(att.delta, 0, None)
y_top = max(float(pos.max()), 1.0) * 1.08
v_top = max(float(pos[k].max()), 1e-6)
extent_l = [0, L.shape[1] * 0.01, 0, 80]
v_lo, v_hi = np.percentile(L, 3), np.percentile(L, 99.7)


def frame(i, final=False):
    band, t_now = order[i]
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    fig.text(0.05, 0.955, "Erase a patch, ask again, keep the drop. Repeat.", fontsize=26, color=INK, **TITLE)
    sub = f"all {len(order)} patches tried once each" if final else f"patch {i + 1} of {len(order)}: band {band + 1} of {B}, {t_now * ps:.1f} to {(t_now + 1) * ps:.1f} s"
    fig.text(0.05, 0.912, f"whisper-base on clip {CLIP}, {sub}", fontsize=15, color=SOFT)
    axm = fig.add_axes([0.08, 0.55, 0.88, 0.30]); axd = fig.add_axes([0.08, 0.30, 0.88, 0.18]); axq = fig.add_axes([0.08, 0.08, 0.88, 0.13])
    axm.imshow(L, origin="lower", aspect="auto", cmap="YlOrBr", extent=extent_l, interpolation="nearest", vmin=v_lo, vmax=v_hi)
    for tt in range(att.n_t + 1):
        axm.axvline(tt * ps, color="white", lw=0.3, alpha=0.5)
    for bb in range(B + 1):
        axm.axhline(bb * mpb, color="white", lw=0.3, alpha=0.5)
    axm.set_xlim(0, dur); axm.set_yticks([0, 40, 80]); axm.set_yticklabels(["0", "1.9 k", "8 k"]); axm.set_ylabel("Hz (mel)")
    axm.tick_params(labelbottom=False)
    if not final:
        axm.add_patch(Rectangle((t_now * ps, band * mpb), ps, mpb, facecolor=PAGE, edgecolor=INK, lw=3))
    axm.set_title("the spectrogram the model hears: one patch erased at a time, row by row from the lowest band",
                  loc="left", color=SOFT, fontsize=14)
    d = pos.max(axis=(1, 2)) if final else pos[:, band, t_now]
    cols = [GREEN if j == k else "#C9BFA8" for j in range(K)]
    axd.bar(np.arange(K), d, color=cols, width=0.8)
    axd.set_ylim(0, y_top); axd.set_xlim(-0.6, K - 0.4); axd.set_xticks(range(K)); axd.set_xticklabels(words, rotation=90, fontsize=11)
    axd.set_ylabel("less sure (nats)")
    axd.set_title("largest drop per word over all patches" if final else "how much less sure the model is of each word with that patch gone",
                  loc="left", color=SOFT, fontsize=14)
    M = np.zeros_like(pos[k])
    if final:
        M = pos[k]
    else:
        M[:band] = pos[k][:band]
        M[band, : t_now + 1] = pos[k][band, : t_now + 1]
    axq.imshow(M, origin="lower", aspect="auto", cmap="Greys", extent=[0, att.n_t * ps, 0, B], interpolation="nearest", vmin=0, vmax=v_top)
    axq.set_xlim(0, dur); axq.set_yticks([0.5, B - 0.5]); axq.set_yticklabels(["low", "high"]); axq.set_xlabel("time (s)")
    axq.set_title(f"the map of '{words[k]}' so far: every patch tried, darker = mattered more", loc="left", color=SOFT, fontsize=14)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


last = frame(len(order) - 1, final=True)
frames = [frame(i) for i in range(len(order))] + [last]
pal = frames[int(np.argmax([pos[k][b, t] for b, t in order]))].quantize(colors=128, method=Image.MEDIANCUT)
frames = [f.quantize(colors=128, palette=pal, dither=Image.NONE) for f in frames]
durations = [FRAME_MS] * len(order) + [FRAME_MS * HOLD_FRAMES]
frames[0].save(IMG / "erase_sweep.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print("gif", len(order), "patches in grid order, written", IMG / "erase_sweep.gif")
