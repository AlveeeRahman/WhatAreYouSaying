"""Build the detector-backbone figure as HTML from experiments/results/detector_backbones.json: panel (a)
dumbbells (flags inside vs outside the known damaged window), panel (b) a dot-and-interval plot of the
noise-focus ratio per speech-to-text model under each detector, with the reference line at 1. Writes
docs/images/src/detector_backbones.html (standalone, for render.py) and the same body as a Design
artboard at the path given as argv[1], if any."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S = json.load(open(ROOT / "experiments/results/detector_backbones.json"))
ASR = ["whisper-tiny", "whisper-base", "moonshine-base"]
COL = {"whisper-tiny": "#2a78d6", "whisper-base": "#c2185b", "moonshine-base": "#1b9e77"}
INK, SOFT, LINE, PAGE, CARD = "#17212E", "#57616F", "#C9C1B3", "#F5F2EB", "#FFFFFF"
XMIN, XMAX = 0.7, 1.25


def pct(v):
    return f"{100 * v:.0f}%"


def xpos(v):
    return f"{100 * (v - XMIN) / (XMAX - XMIN):.1f}%"


names = list(S["detectors"])
ref_left = xpos(1.0)
short = {n: n.split(" (")[0] + (" · AADD v3.1" if "AADD" in n else " · trained here") for n in names}

# ---- panel (a): dumbbells --------------------------------------------------------------------------
rows_a = []
for n in names:
    o = S["controlled"][n]["overall"]
    i, out = o["flagged_inside"], o["flagged_outside"]
    rows_a.append(f'''
<div style="display: grid; grid-template-columns: 150px 1fr 132px; align-items: center; gap: 14px; height: 48px;">
  <div style="font-size: 15px; color: {INK}; line-height: 20px;">{short[n]}</div>
  <div style="position: relative; height: 24px;">
    <div style="position: absolute; left: 0; right: 0; top: 11px; height: 2px; background: {LINE};"></div>
    <div style="position: absolute; left: {100 * out:.1f}%; width: {100 * (i - out):.1f}%; top: 9px; height: 6px; background: #D9D3C7; border-radius: 3px;"></div>
    <div style="position: absolute; left: calc({100 * out:.1f}% - 8px); top: 4px; width: 16px; height: 16px; border-radius: 50%; background: #c2185b; border: 2px solid {CARD};"></div>
    <div style="position: absolute; left: calc({100 * i:.1f}% - 8px); top: 4px; width: 16px; height: 16px; border-radius: 50%; background: #2a78d6; border: 2px solid {CARD};"></div>
  </div>
  <div style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {SOFT};"><span style="color: #2a78d6; font-weight: 600;">{pct(i)}</span> / <span style="color: #c2185b; font-weight: 600;">{pct(out)}</span></div>
</div>''')

# per-class hit rate (right class flagged inside its window), one cell per damage kind and detector
CLS = ["noise", "reverb", "clipping", "lowpass", "hum", "clicks"]
def cell(v):
    a = 0.12 + 0.75 * v
    return f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 13px; text-align: center; padding: 6px 0; border-radius: 6px; background: rgba(42,120,214,{a:.2f}); color: {"#FFFFFF" if v >= 0.55 else INK};">{100 * v:.0f}%</div>'
head = ''.join(f'<div style="font-size: 12px; color: {SOFT}; text-align: center; line-height: 15px;">{n.split(" (")[0]}</div>' for n in names)
table = ''.join(f'<div style="font-size: 13px; color: {INK};">{c}</div>' + ''.join(cell(S["controlled"][n]["by_class"][c]["class_inside"]) for n in names) for c in CLS)
by_class = f'''
<div style="margin-top: 10px; font-size: 14px; font-weight: 600;">Right kind of damage flagged inside its window, per kind</div>
<div style="display: grid; grid-template-columns: 90px repeat(3, minmax(0, 1fr)); gap: 6px 10px; align-items: center;"><div></div>{head}{table}</div>'''

# ---- panel (b): dot and interval -------------------------------------------------------------------
groups_b = []
for n in names:
    rows = []
    for m in ASR:
        d = S["focus"][n][m]
        lo, hi, mean = d["ci"][0], d["ci"][1], d["mean"]
        note = "" if lo <= 1 <= hi else f'<span style="color: #c2185b; font-weight: 600;"> below 1</span>'
        rows.append(f'''
<div style="display: grid; grid-template-columns: 118px 1fr 232px; align-items: center; gap: 12px; height: 34px;">
  <div style="font-size: 14px; color: {INK};">{m}</div>
  <div style="position: relative; height: 22px;">
    <div style="position: absolute; left: {ref_left}; top: -6px; bottom: -6px; border-left: 2px dashed #9AA3AE;"></div>
    <div style="position: absolute; left: {xpos(lo)}; width: calc({xpos(hi)} - {xpos(lo)}); top: 9px; height: 4px; background: {COL[m]}; opacity: 0.45; border-radius: 2px;"></div>
    <div style="position: absolute; left: calc({xpos(mean)} - 7px); top: 4px; width: 14px; height: 14px; border-radius: 50%; background: {COL[m]}; border: 2px solid {CARD};"></div>
  </div>
  <div style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK};">{mean:.2f} <span style="color: {SOFT};">[{lo:.2f}, {hi:.2f}]</span>{note}</div>
</div>''')
    groups_b.append(f'''
<div style="display: flex; flex-direction: column; gap: 2px; padding: 10px 0 6px;">
  <div style="font-size: 15px; font-weight: 600; color: {INK}; margin-bottom: 4px;">{short[n]}</div>
  {''.join(rows)}
</div>''')

ticks = "".join(f'<div style="position: absolute; left: {xpos(v)}; top: 0; transform: translateX(-50%); font-size: 12px; color: {SOFT};">{v:.1f}</div>'
                for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2))
body = f'''
<div style="width: 1200px; height: 830px; box-sizing: border-box; padding: 40px 48px; background: {PAGE}; color: {INK}; font-family: 'IBM Plex Sans', system-ui, sans-serif; display: flex; flex-direction: column; gap: 18px;">
  <div style="display: flex; flex-direction: column; gap: 6px;">
    <h1 style="margin: 0; font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 30px; line-height: 36px; font-weight: 700;">Same question, three detector backbones</h1>
    <p style="margin: 0; font-size: 15px; line-height: 22px; color: {SOFT}; max-width: 1000px;">The attribution maps never change. Only the detector that says which audio is degraded does: AADD's own (whisper-small inside), and two trained here with AADD's recipe on the whisper-base and Moonshine encoders.</p>
  </div>
  <div style="display: grid; grid-template-columns: 500px 1fr; gap: 28px; flex: 1;">
    <div style="background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 20px 22px; display: flex; flex-direction: column; gap: 10px;">
      <div style="font-size: 17px; font-weight: 600;">(a) Does the detector find the damage?</div>
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px;">60 LibriSpeech clips with one known damaged 2 s each. Share of 20 ms frames flagged <span style="color: #2a78d6; font-weight: 600;">inside</span> the window and <span style="color: #c2185b; font-weight: 600;">outside</span> it.</div>
      <div style="display: grid; grid-template-columns: 150px 1fr 132px; gap: 14px; font-size: 12px; color: {SOFT};"><div></div><div style="display: flex; justify-content: space-between;"><span>0%</span><span>50%</span><span>100%</span></div><div></div></div>
      {''.join(rows_a)}
      {by_class}
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px; margin-top: auto;">The wider the gap in the dumbbells, the better the detector separates damaged from clean audio. AADD's own detector is the strongest; the Moonshine one misses most clicks and hum.</div>
    </div>
    <div style="background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 20px 22px; display: flex; flex-direction: column; gap: 6px;">
      <div style="font-size: 17px; font-weight: 600;">(b) Does any model lean on degraded audio?</div>
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px;">100 VoxCeleb clips. Noise-focus ratio = evidence on flagged audio ÷ share of the clip flagged. Dot = mean over clips, bar = 95% bootstrap interval. 1.0 means no preference.</div>
      <div style="display: grid; grid-template-columns: 118px 1fr 232px; gap: 12px;"><div></div><div style="position: relative; height: 16px;">{ticks}</div><div style="font-size: 12px; color: {SOFT};">dashed line = 1.0, no preference</div></div>
      <div style="position: relative;">
        {''.join(groups_b)}
      </div>
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px; margin-top: auto;">Under the two detectors that find the damage, every interval contains 1. The weakest detector flags frames the models do not use, so its ratios fall below 1: a detector floor, not a model preference.</div>
    </div>
  </div>
</div>'''

standalone = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Three detector backbones</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>html, body {{ margin: 0; background: {PAGE}; }}</style></head>
<body>{body}
<script>
const want = ["700 30px 'Space Grotesk'", "400 15px 'IBM Plex Sans'", "400 13px 'IBM Plex Mono'"];
Promise.all(want.map(f => document.fonts.load(f))).then(() => {{ document.body.dataset.overflow = ""; document.body.dataset.fonts = want.map(f => document.fonts.check(f) ? 'ok' : 'MISSING').join(','); }});
</script></body></html>'''
(HERE / "detector_backbones.html").write_text(standalone)

artboard = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Three detector backbones</title>
<script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
body{{margin:0}}
a{{color:#2F5FD0}}a:hover{{color:#17212E}}
</style>
</helmet>{body}
</x-dc>
<script type="text/x-dc" data-dc-script data-props='{{"$preview":{{"width":1200,"height":830}}}}'>
class Component extends DCLogic {{
renderVals() {{
return {{}};
}}
}}
</script>
</body>
</html>'''
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(artboard)
print("written", HERE / "detector_backbones.html")
