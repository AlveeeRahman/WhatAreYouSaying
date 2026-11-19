"""Build the VoiceBank-DEMAND tracking figure as HTML from experiments/results/demand/summary.json:
panel (a) dot-and-interval per model for the share of words whose evidence sits on the right side
(noisy or clean) and in the very segment where the word is spoken, each against its chance level;
panel (b) dumbbells: mass of a word's evidence in noisy audio when the word is spoken in noise versus
in clean audio, and AADD's flags in noisy versus clean segments. Writes docs/images/src/demand_tracking.html
(standalone, for render.py)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S = json.load(open(ROOT / "experiments/results/demand/summary.json"))
ASR = [m for m in ("whisper-tiny", "whisper-base", "moonshine-base") if m in S["models"] and "agree" in S["models"][m]]
COL = {"whisper-tiny": "#2a78d6", "whisper-base": "#c2185b", "moonshine-base": "#1b9e77"}
INK, SOFT, LINE, PAGE, CARD = "#17212E", "#57616F", "#C9C1B3", "#F5F2EB", "#FFFFFF"
XMIN, XMAX = 0.0, 1.0


def pct(v):
    return f"{100 * v:.0f}%"


def xpos(v):
    return f"{100 * (v - XMIN) / (XMAX - XMIN):.1f}%"


def dot_rows(key, chance_key):
    rows = []
    for m in ASR:
        d, c = S["models"][m][key], S["models"][m][chance_key]["mean"]
        lo, hi, mean = d["ci"][0], d["ci"][1], d["mean"]
        rows.append(f'''
<div style="display: grid; grid-template-columns: 118px 1fr 250px; align-items: center; gap: 12px; height: 34px;">
  <div style="font-size: 14px; color: {INK};">{m}</div>
  <div style="position: relative; height: 22px;">
    <div style="position: absolute; left: {xpos(c)}; top: -6px; bottom: -6px; border-left: 2px dashed #9AA3AE;"></div>
    <div style="position: absolute; left: {xpos(lo)}; width: calc({xpos(hi)} - {xpos(lo)}); top: 9px; height: 4px; background: {COL[m]}; opacity: 0.45; border-radius: 2px;"></div>
    <div style="position: absolute; left: calc({xpos(mean)} - 7px); top: 4px; width: 14px; height: 14px; border-radius: 50%; background: {COL[m]}; border: 2px solid {CARD};"></div>
  </div>
  <div style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK};">{pct(mean)} <span style="color: {SOFT};">[{pct(lo)}, {pct(hi)}]</span> <span style="color: {SOFT};">chance {pct(c)}</span></div>
</div>''')
    return "".join(rows)


def dumbbell(label, lo_v, hi_v, lo_col, hi_col):
    return f'''
<div style="display: grid; grid-template-columns: 150px 1fr 132px; align-items: center; gap: 14px; height: 44px;">
  <div style="font-size: 14px; color: {INK}; line-height: 18px;">{label}</div>
  <div style="position: relative; height: 24px;">
    <div style="position: absolute; left: 0; right: 0; top: 11px; height: 2px; background: {LINE};"></div>
    <div style="position: absolute; left: {100 * min(lo_v, hi_v):.1f}%; width: {100 * abs(hi_v - lo_v):.1f}%; top: 9px; height: 6px; background: #D9D3C7; border-radius: 3px;"></div>
    <div style="position: absolute; left: calc({100 * lo_v:.1f}% - 8px); top: 4px; width: 16px; height: 16px; border-radius: 50%; background: {lo_col}; border: 2px solid {CARD};"></div>
    <div style="position: absolute; left: calc({100 * hi_v:.1f}% - 8px); top: 4px; width: 16px; height: 16px; border-radius: 50%; background: {hi_col}; border: 2px solid {CARD};"></div>
  </div>
  <div style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {SOFT};"><span style="color: {hi_col}; font-weight: 600;">{pct(hi_v)}</span> / <span style="color: {lo_col}; font-weight: 600;">{pct(lo_v)}</span></div>
</div>'''


ticks = "".join(f'<div style="position: absolute; left: {xpos(v)}; top: 0; transform: translateX(-50%); font-size: 12px; color: {SOFT};">{pct(v)}</div>'
                for v in (0, 0.25, 0.5, 0.75, 1.0))
axis = f'<div style="display: grid; grid-template-columns: 118px 1fr 250px; gap: 12px;"><div></div><div style="position: relative; height: 16px;">{ticks}</div><div style="font-size: 12px; color: {SOFT}; padding-left: 16px;">dashed line = chance</div></div>'
mass_rows = "".join(dumbbell(m, S["models"][m]["mass_noisy_when_spoken_clean"]["mean"], S["models"][m]["mass_noisy_when_spoken_noisy"]["mean"], "#c2185b", "#2a78d6") for m in ASR)
a = S["aadd"]["flagged"]
aadd_row = dumbbell("AADD v3 flags", a["alt_flagged_in_clean"], a["alt_flagged_in_noisy"], "#c2185b", "#2a78d6")
wer = "".join(f'<div style="font-size: 13px; color: {INK};">{m}</div>' + "".join(f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 13px; text-align: center;">{pct(S["models"][m]["wer"][c]["mean"])}</div>' for c in ("clean", "noisy", "alt")) for m in ASR)
n = S["design"]["n_clips"]
body = f'''
<div style="width: 1200px; height: 830px; box-sizing: border-box; padding: 40px 48px; background: {PAGE}; color: {INK}; font-family: 'IBM Plex Sans', system-ui, sans-serif; display: flex; flex-direction: column; gap: 18px;">
  <div style="display: flex; flex-direction: column; gap: 6px;">
    <h1 style="margin: 0; font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 30px; line-height: 36px; font-weight: 700;">Noise that comes and goes: does each word land on the right audio?</h1>
    <p style="margin: 0; font-size: 15px; line-height: 22px; color: {SOFT}; max-width: 1040px;">{n} VoiceBank-DEMAND sentences (CC BY 4.0), real bus, cafe, living-room, office and square noise at 2.5 to 17.5 dB, switched on and off in 4 to 7 segments per sentence. The truth for where each word is spoken comes from the clean recording. Dot = mean over clips, bar = 95% bootstrap interval.</p>
  </div>
  <div style="display: grid; grid-template-columns: 1fr 470px; gap: 28px; flex: 1;">
    <div style="background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 20px 22px; display: flex; flex-direction: column; gap: 6px;">
      <div style="font-size: 17px; font-weight: 600;">(a) Where sound-token places each word</div>
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px;">Share of words whose strongest patch of evidence sits on the same kind of audio the word was spoken on (noisy or clean), and in the very segment it was spoken in.</div>
      {axis}
      <div style="font-size: 14px; font-weight: 600; margin-top: 6px;">Right side, noisy or clean</div>
      {dot_rows("agree", "chance_agree")}
      <div style="font-size: 14px; font-weight: 600; margin-top: 8px;">Right segment</div>
      {dot_rows("same_segment", "chance_same_segment")}
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px; margin-top: auto;">Chance is what a randomly placed patch would score given each clip's noisy share and segment lengths.</div>
    </div>
    <div style="background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 20px 22px; display: flex; flex-direction: column; gap: 10px;">
      <div style="font-size: 17px; font-weight: 600;">(b) Evidence in noisy audio</div>
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px;">Share of a word's evidence that lies in the noisy segments when the word is spoken <span style="color: #2a78d6; font-weight: 600;">in noise</span> and when it is spoken <span style="color: #c2185b; font-weight: 600;">in clean audio</span>. Last row: share of 20 ms frames AADD flags in the noisy and the clean segments.</div>
      <div style="display: grid; grid-template-columns: 150px 1fr 132px; gap: 14px; font-size: 12px; color: {SOFT};"><div></div><div style="display: flex; justify-content: space-between;"><span>0%</span><span>50%</span><span>100%</span></div><div></div></div>
      {mass_rows}
      {aadd_row}
      <div style="margin-top: 10px; font-size: 14px; font-weight: 600;">Word error rate per condition</div>
      <div style="display: grid; grid-template-columns: 118px repeat(3, minmax(0, 1fr)); gap: 6px 10px; align-items: center;"><div></div><div style="font-size: 12px; color: {SOFT}; text-align: center;">clean</div><div style="font-size: 12px; color: {SOFT}; text-align: center;">noisy throughout</div><div style="font-size: 12px; color: {SOFT}; text-align: center;">on and off</div>{wer}</div>
      <div style="font-size: 13px; color: {SOFT}; line-height: 18px; margin-top: auto;">The attribution follows the words, not the noise: a word spoken in noise draws its evidence from the noisy stretch, a word spoken in clean audio does not. AADD's flags bleed into the clean segments next to the noise.</div>
    </div>
  </div>
</div>'''
standalone = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Noise that comes and goes</title>
<style>
@font-face {{ font-family: 'Space Grotesk'; font-weight: 300 700; src: url('../fonts/SpaceGrotesk[wght].ttf'); }}
@font-face {{ font-family: 'IBM Plex Sans'; font-weight: 100 700; src: url('../fonts/IBMPlexSans[wdth,wght].ttf'); }}
@font-face {{ font-family: 'IBM Plex Mono'; font-weight: 400; src: url('../fonts/IBMPlexMono-Regular.ttf'); }}
@font-face {{ font-family: 'IBM Plex Mono'; font-weight: 600; src: url('../fonts/IBMPlexMono-SemiBold.ttf'); }}
html, body {{ margin: 0; background: {PAGE}; }}</style></head>
<body>{body}
<script>
const want = ["700 30px 'Space Grotesk'", "400 15px 'IBM Plex Sans'", "400 13px 'IBM Plex Mono'"];
Promise.all(want.map(f => document.fonts.load(f))).then(() => {{ document.body.dataset.overflow = ""; document.body.dataset.fonts = want.map(f => document.fonts.check(f) ? 'ok' : 'MISSING').join(','); }});
</script></body></html>'''
(HERE / "demand_tracking.html").write_text(standalone)
print("written", HERE / "demand_tracking.html")
