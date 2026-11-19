"""The WAYS icon: a big ear with a hand cupped behind it, three sound arcs coming in. Drawn with
matplotlib paths so it needs no extra tools; writes ways_icon.svg and ways_icon.png next to itself."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as P

HERE = Path(__file__).resolve().parent
SKIN, LINE, BLUE, BG = "#f0bd7a", "#5a3410", "#256abf", "#fcfcfb"
HAND, HAND_DARK, EAR, EAR_LIGHT, EAR_DARK = "#d49a52", "#b9803f", "#f7d39a", "#fbe6c0", "#dca764"
M, L, C4, Z = P.MOVETO, P.LINETO, P.CURVE4, P.CLOSEPOLY


def bez(points):
    """points: [(x, y), (c1, c2, p), ...] -> closed Path."""
    verts, codes = [points[0]], [M]
    for seg in points[1:]:
        verts += list(seg); codes += [C4, C4, C4]
    verts.append(points[0]); codes.append(Z)
    return P(verts, codes)


def bez_open(points):
    verts, codes = [points[0]], [M]
    for seg in points[1:]:
        verts += list(seg); codes += [C4, C4, C4]
    return P(verts, codes)


fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((50, 50), 49, facecolor=BG, edgecolor="#e4e3df", lw=1.5))

# --- hand, cupped behind the ear (drawn first so the ear sits in front of the palm) -------------
palm = bez([(58, 22), ((52, 30), (50, 48), (56, 62)), ((60, 72), (72, 80), (84, 76)), ((94, 72), (96, 50), (90, 34)),
            ((86, 24), (70, 16), (58, 22))])
ax.add_patch(PathPatch(palm, facecolor=HAND, edgecolor=LINE, lw=2.2, joinstyle="round"))
# four fingers fanning up and over the top of the ear, slightly curled to the left
fingers = [
    bez([(56, 60), ((50, 72), (48, 84), (54, 90)), ((58, 94), (64, 92), (64, 86)), ((64, 78), (66, 70), (68, 64)), ((64, 62), (60, 61), (56, 60))]),
    bez([(66, 64), ((62, 76), (62, 88), (68, 93)), ((72, 97), (78, 94), (77, 88)), ((76, 80), (78, 72), (79, 66)), ((75, 64), (70, 63), (66, 64))]),
    bez([(78, 66), ((76, 76), (78, 86), (84, 90)), ((88, 93), (93, 89), (91, 83)), ((89, 76), (90, 70), (90, 65)), ((86, 64), (82, 64), (78, 66))]),
    bez([(88, 62), ((89, 70), (92, 76), (95, 75)), ((98, 74), (98, 69), (96, 65)), ((94, 60), (94, 54), (93, 48)), ((91, 50), (89, 56), (88, 62))]),
]
for f in fingers:
    ax.add_patch(PathPatch(f, facecolor=HAND, edgecolor=LINE, lw=2.2, joinstyle="round"))
# thumb along the bottom, tucked under the lobe
thumb = bez([(64, 26), ((58, 16), (50, 10), (44, 13)), ((39, 16), (41, 23), (47, 27)), ((53, 31), (60, 32), (64, 26))])
ax.add_patch(PathPatch(thumb, facecolor=HAND, edgecolor=LINE, lw=2.2, joinstyle="round"))

# --- the ear ----------------------------------------------------------------------------------
outer = bez([(36, 70), ((36, 84), (48, 92), (58, 88)), ((70, 84), (74, 70), (70, 56)), ((67, 46), (62, 42), (58, 32)),
             ((55, 22), (46, 18), (40, 24)), ((34, 30), (36, 40), (38, 48)), ((38, 56), (36, 62), (36, 70))])
from matplotlib.transforms import Affine2D

shadow = Affine2D().translate(2.2, -2.2).transform_path(outer)
ax.add_patch(PathPatch(shadow, facecolor=HAND_DARK, edgecolor="none", alpha=0.6))
ax.add_patch(PathPatch(outer, facecolor=EAR, edgecolor=LINE, lw=2.6, joinstyle="round"))
concha = bez([(44, 40), ((42, 52), (46, 66), (56, 68)), ((62, 69), (64, 60), (60, 54)), ((56, 48), (50, 40), (44, 40))])
ax.add_patch(PathPatch(concha, facecolor=EAR_DARK, edgecolor="none", alpha=0.7))
rim = bez([(40, 66), ((42, 80), (52, 86), (58, 82)), ((66, 77), (67, 66), (62, 58)), ((64, 66), (62, 76), (56, 79)), ((50, 82), (44, 76), (44, 66))])
ax.add_patch(PathPatch(rim, facecolor=EAR_LIGHT, edgecolor="none", alpha=0.9))
helix = bez_open([(40, 68), ((42, 80), (52, 86), (58, 82)), ((65, 77), (66, 66), (62, 58))])
antihelix = bez_open([(46, 44), ((44, 56), (50, 70), (58, 70)), ((63, 70), (63, 60), (58, 56))])
tragus = bez_open([(40, 50), ((45, 50), (47, 44), (44, 40))])
canal = bez_open([(50, 52), ((54, 56), (58, 52), (54, 46))])
for pth, lw in ((helix, 2.2), (antihelix, 2.2), (tragus, 2.0), (canal, 1.8)):
    ax.add_patch(PathPatch(pth, facecolor="none", edgecolor=LINE, lw=lw, capstyle="round"))

# --- sound coming in from the left: tapered crescents, fat in the middle, with a lighter highlight ---
import numpy as np
from matplotlib.patches import Polygon

BLUE_DARK, BLUE_LIGHT = "#174f9c", "#8fbdf0"
cx, cy = 40, 52
for r, half, span in ((13, 2.6, 0.26), (21, 3.1, 0.28), (29, 3.6, 0.30)):
    t = np.linspace(np.pi * (1 - span), np.pi * (1 + span), 90)
    bulge = np.sin((t - t[0]) / (t[-1] - t[0]) * np.pi) ** 0.7          # 0 at the ends, 1 in the middle
    w = 0.6 + (half - 0.6) * bulge
    outer = np.stack([cx + (r + w) * np.cos(t), cy + (r + w) * np.sin(t)], 1)
    inner = np.stack([cx + (r - w) * np.cos(t), cy + (r - w) * np.sin(t)], 1)
    # soft shadow, then the body, then a highlight along the outer edge
    ax.add_patch(Polygon(np.vstack([outer, inner[::-1]]) + [0.7, -0.7], closed=True, facecolor=BLUE_DARK, edgecolor="none", alpha=0.35))
    ax.add_patch(Polygon(np.vstack([outer, inner[::-1]]), closed=True, facecolor=BLUE, edgecolor=BLUE_DARK, lw=1.2, joinstyle="round"))
    hw = w * 0.38
    h_out = np.stack([cx + (r + w - 0.9) * np.cos(t), cy + (r + w - 0.9) * np.sin(t)], 1)
    h_in = np.stack([cx + (r + w - 0.9 - hw) * np.cos(t), cy + (r + w - 0.9 - hw) * np.sin(t)], 1)
    ax.add_patch(Polygon(np.vstack([h_out, h_in[::-1]]), closed=True, facecolor=BLUE_LIGHT, edgecolor="none", alpha=0.75))

fig.savefig(HERE / "ways_icon.png", dpi=100, transparent=True)
fig.savefig(HERE / "ways_icon.svg", transparent=True)
print("icon written")
