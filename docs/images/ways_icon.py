"""The WAYS icon: a big ear with a hand cupped behind it, three sound arcs coming in.

Vector drawing with matplotlib paths, no extra tools. Fingers and thumb are built from a centreline
Bezier and a width profile (wide at the base, narrower at the tip, rounded end), so their outlines
are smooth and consistent; the palm, wrist and ear are closed cubic Bezier shapes. Writes
ways_icon.svg and ways_icon.png next to this file.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon
from matplotlib.path import Path as P

HERE = Path(__file__).resolve().parent
SKIN, SKIN_DARK, SKIN_LIGHT, LINE, BLUE, BG, NAIL = "#f2c27f", "#dea35d", "#f8d7a6", "#5a3410", "#256abf", "#fcfcfb", "#fbe3c4"
M, C4, Z = P.MOVETO, P.CURVE4, P.CLOSEPOLY


def closed(points):
    verts, codes = [points[0]], [M]
    for seg in points[1:]:
        verts += list(seg); codes += [C4, C4, C4]
    verts.append(points[0]); codes.append(Z)
    return P(verts, codes)


def opened(points):
    verts, codes = [points[0]], [M]
    for seg in points[1:]:
        verts += list(seg); codes += [C4, C4, C4]
    return P(verts, codes)


def bezier(p0, c1, c2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    p0, c1, c2, p3 = map(np.asarray, (p0, c1, c2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3


def digit(p0, c1, c2, p3, w0, w1, n=80):
    """Outline polygon of a finger: centreline Bezier offset by a width that tapers from w0 to w1,
    closed with a semicircular tip and a flat base (the base is hidden in the palm)."""
    c = bezier(p0, c1, c2, p3, n)
    d = np.gradient(c, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, n)[:, None]
    w = (w0 + (w1 - w0) * t ** 0.8) / 2
    left, right = c + nrm * w, c - nrm * w
    tip_c, tip_d, r = c[-1], d[-1], float(w[-1, 0])
    ang = np.arctan2(nrm[-1, 1], nrm[-1, 0])
    a = np.linspace(ang, ang - np.pi, 24)
    tip = np.stack([tip_c[0] + r * np.cos(a), tip_c[1] + r * np.sin(a)], 1)
    return np.vstack([left, tip, right[::-1]]), c, d, w[:, 0]


def crease(ax, c, d, w, frac, depth=0.45):
    """A short knuckle crease across the finger at fraction `frac` of its length, bowed toward the tip."""
    i = int(frac * (len(c) - 1))
    nrm = np.array([-d[i, 1], d[i, 0]])
    a, b = c[i] + nrm * w[i] * 0.75, c[i] - nrm * w[i] * 0.75
    mid = c[i] + d[i] * w[i] * depth
    ax.add_patch(PathPatch(opened([tuple(a), (tuple(a * 0.5 + mid * 0.5), tuple(b * 0.5 + mid * 0.5), tuple(b))]),
                           facecolor="none", edgecolor=LINE, lw=1.3, capstyle="round", alpha=0.85))


def nail(ax, c, d, w):
    tip, dirn = c[-1], d[-1]
    centre = tip - dirn * w[-1] * 0.95
    ang = np.degrees(np.arctan2(dirn[1], dirn[0]))
    ax.add_patch(Ellipse(centre, w[-1] * 1.0, w[-1] * 1.3, angle=ang, facecolor=NAIL, edgecolor=LINE, lw=1.1))


fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((50, 50), 49, facecolor=BG, edgecolor="#e4e3df", lw=1.5))
clip = Circle((50, 50), 48.5, transform=ax.transData)

# ---------------------------------------------------------------- hand (behind the ear)
# wrist, running off the bottom right of the circle
wrist = Polygon([(66, 28), (88, 31), (95, 8), (76, 0)], closed=True, facecolor=SKIN, edgecolor=LINE, lw=2.2, joinstyle="round")
wrist.set_clip_path(clip); ax.add_patch(wrist)
# palm: rounded knuckle row on top, rounded heel into the wrist
palm = closed([(57, 60), ((52, 52), (53, 38), (59, 30)), ((65, 24), (82, 24), (89, 31)), ((94, 37), (94, 50), (90, 57)),
               ((84, 64), (68, 66), (57, 60))])
ax.add_patch(PathPatch(palm, facecolor=SKIN, edgecolor=LINE, lw=2.2, joinstyle="round"))
shade = closed([(62, 27), ((70, 24), (84, 25), (89, 32)), ((93, 38), (93, 46), (91, 51)), ((86, 44), (74, 35), (62, 27))])
ax.add_patch(PathPatch(shade, facecolor=SKIN_DARK, edgecolor="none", alpha=0.5))
# fingers, little to index so the nearer ones overlap the farther ones; bases touch along the knuckle row
specs = [  # p0 (base), c1, c2, p3 (tip), base width, tip width
    ((88, 55), (92, 62), (96, 68), (96, 74), 9.0, 7.0),       # little
    ((80, 61), (82, 72), (85, 81), (84, 87), 10.5, 8.0),      # ring
    ((70, 63), (70, 75), (69, 85), (68, 92), 11.0, 8.5),      # middle
    ((60, 61), (57, 72), (53, 81), (50, 87), 10.5, 8.0),      # index, curling over the ear
]
for p0, c1, c2, p3, w0, w1 in specs:
    poly, c, d, w = digit(p0, c1, c2, p3, w0, w1)
    ax.add_patch(Polygon(poly, closed=True, facecolor=SKIN, edgecolor=LINE, lw=2.2, joinstyle="round"))
    crease(ax, c, d, w, 0.45); crease(ax, c, d, w, 0.75)
    nail(ax, c, d, w)
# thumb, short and angled down under the lobe
poly, c, d, w = digit((60, 33), (55, 29), (49, 24), (44, 21), 10.0, 7.5)
ax.add_patch(Polygon(poly, closed=True, facecolor=SKIN, edgecolor=LINE, lw=2.2, joinstyle="round"))
crease(ax, c, d, w, 0.5); nail(ax, c, d, w)

# ---------------------------------------------------------------- the ear (in front)
outer = closed([(36, 70), ((36, 84), (48, 92), (58, 88)), ((70, 84), (74, 70), (70, 56)), ((67, 46), (62, 42), (58, 32)),
                ((55, 22), (46, 18), (40, 24)), ((34, 30), (36, 40), (38, 48)), ((38, 56), (36, 62), (36, 70))])
ax.add_patch(PathPatch(outer, facecolor=SKIN, edgecolor=LINE, lw=2.6, joinstyle="round"))
concha = closed([(44, 40), ((42, 52), (46, 66), (56, 68)), ((62, 69), (64, 60), (60, 54)), ((56, 48), (50, 40), (44, 40))])
ax.add_patch(PathPatch(concha, facecolor=SKIN_DARK, edgecolor="none", alpha=0.45))
rim = closed([(40, 66), ((42, 80), (52, 86), (58, 82)), ((66, 77), (67, 66), (62, 58)), ((64, 66), (62, 76), (56, 79)), ((50, 82), (44, 76), (44, 66))])
ax.add_patch(PathPatch(rim, facecolor=SKIN_LIGHT, edgecolor="none", alpha=0.8))
helix = opened([(40, 68), ((42, 80), (52, 86), (58, 82)), ((65, 77), (66, 66), (62, 58))])
antihelix = opened([(46, 44), ((44, 56), (50, 70), (58, 70)), ((63, 70), (63, 60), (58, 56))])
tragus = opened([(40, 50), ((45, 50), (47, 44), (44, 40))])
canal = opened([(50, 52), ((54, 56), (58, 52), (54, 46))])
for pth, lw in ((helix, 2.2), (antihelix, 2.2), (tragus, 2.0), (canal, 1.8)):
    ax.add_patch(PathPatch(pth, facecolor="none", edgecolor=LINE, lw=lw, capstyle="round"))

# ---------------------------------------------------------------- sound coming in from the left
for r, lw in ((13, 3.2), (20, 2.8), (27, 2.4)):
    t = np.linspace(np.pi * 0.72, np.pi * 1.28, 60)
    ax.plot(40 + r * np.cos(t), 52 + r * np.sin(t), color=BLUE, lw=lw, solid_capstyle="round")

fig.savefig(HERE / "ways_icon.png", dpi=100, transparent=True)
fig.savefig(HERE / "ways_icon.svg", transparent=True)
print("icon written")
