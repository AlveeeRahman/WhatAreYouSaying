"""A GIF of the method on real data: a patch sweeps along one pitch band of the example clip's spectrogram,
and for each position the bars show how much less sure whisper-base became of every word. The map of the
word that reacts most builds up underneath. Drawn in the hand-drawn figure style (cream ground,
Kalam and Patrick Hand). Writes docs/images/erase_sweep.gif. Usage: python experiments/gif_erase_sweep.py [data folder]"""
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
k = 13                                             # ' Ron', the word the method figure uses
band = int(np.unravel_index(att.delta[k].argmax(), att.delta[k].shape)[0])
words = [p.strip() or "_" for p in att.pieces]
K = len(words)


def frame(t_now, final=False):
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    fig.text(0.05, 0.955, "Erase a patch, ask again, keep the drop. Repeat.", fontsize=26, color=INK, **TITLE)
    fig.text(0.05, 0.912, f"whisper-base on clip {CLIP}, band {band} of 8, sweeping left to right", fontsize=15, color=SOFT)
    axm = fig.add_axes([0.08, 0.55, 0.88, 0.30]); axd = fig.add_axes([0.08, 0.30, 0.88, 0.18]); axq = fig.add_axes([0.08, 0.08, 0.88, 0.13])
    axm.imshow(L, origin="lower", aspect="auto", cmap="YlOrBr", extent=[0, L.shape[1] * 0.01, 0, 80], interpolation="nearest",
               vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.7))
    for tt in range(att.n_t + 1):
        axm.axvline(tt * ps, color="white", lw=0.3, alpha=0.5)
    for bb in range(B + 1):
        axm.axhline(bb * mpb, color="white", lw=0.3, alpha=0.5)
    axm.set_xlim(0, dur); axm.set_yticks([0, 40, 80]); axm.set_yticklabels(["0", "1.9 k", "8 k"]); axm.set_ylabel("Hz (mel)")
    axm.tick_params(labelbottom=False)
    if not final:
        axm.add_patch(Rectangle((t_now * ps, band * mpb), ps, mpb, facecolor=PAGE, edgecolor=PURPLE, lw=3))
        axm.text(t_now * ps + ps / 2, band * mpb + mpb + 1, "erased", color=PURPLE, ha="center", va="bottom", fontsize=14, **TITLE)
    axm.set_title("the spectrogram the model hears, with the 100 ms x 1 band patch grid", loc="left", color=SOFT, fontsize=14)
    d = att.delta[:, band, t_now] if not final else att.delta[:, band, :].max(1)
    cols = [GREEN if i == k else "#C9BFA8" for i in range(K)]
    axd.bar(np.arange(K), np.clip(d, 0, None), color=cols, width=0.8)
    axd.set_ylim(0, max(att.delta[:, band, :].max(), 1) * 1.08); axd.set_xticks(range(K)); axd.set_xticklabels(words, rotation=90, fontsize=11)
    axd.set_ylabel("less sure (nats)")
    axd.set_title("how much less sure the model is of each word with that patch gone" if not final else "largest drop per word over the whole band",
                  loc="left", color=SOFT, fontsize=14)
    M = np.zeros_like(att.delta[k]); M[:, : t_now + 1] = np.clip(att.delta[k][:, : t_now + 1], 0, None)
    if final:
        M = np.clip(att.delta[k], 0, None)
    axq.imshow(M, origin="lower", aspect="auto", cmap="Greys", extent=[0, att.n_t * ps, 0, B], interpolation="nearest", vmin=0, vmax=max(att.delta[k].max(), 1e-6))
    axq.set_xlim(0, dur); axq.set_yticks([0.5, B - 0.5]); axq.set_yticklabels(["low", "high"]); axq.set_xlabel("time (s)")
    axq.set_title(f"the map of '{words[k]}' so far: every patch tried, darker = mattered more", loc="left", color=SOFT, fontsize=14)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = [frame(t) for t in range(att.n_t)] + [frame(att.n_t - 1, final=True)] * 12
pal = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
frames = [f.quantize(colors=128, palette=pal, dither=Image.NONE) for f in frames]
durations = [140] * att.n_t + [140] * 12
frames[0].save(IMG / "erase_sweep.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print("gif", att.n_t, "sweep frames, written", IMG / "erase_sweep.gif")
