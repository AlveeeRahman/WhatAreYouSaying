"""One ambiguous clip read in depth: the LibriSpeech sentence with pink noise over its middle two seconds
(the controlled experiment's noise example), where whisper-base hears "fire and sprite told" and
moonshine-base "lions, fight cold" for the gold "my iron's twite told".

Three readings of the same tokens, for each model and for the degraded and the clean recording:
  erasure      sound-token: the patch each token used (ways.attribution.explain)
  grad x act   aud-cam with tokens=True: gradient times activation per frame, per token and layer
  Grad-CAM     the same with the time-pooled channel weights of Grad-CAM proper (ways.latent.token_cam pooled=True)
Each reading places every token in time; the tokens are in spoken order, so a reading that works gives a
Spearman correlation near 1 between token index and placed time. Then, per token, the erasure test and the
gradient map are read against each other on the noisy stretch, and the clean recording is the control: the
words after the noise should keep their anchor.

Usage: python experiments/ambiguous_case.py            (needs experiments/cache/controlled from controlled.py)
       python experiments/ambiguous_case.py --summary  (tables and the three-model figure, from the saved results)

Also runs the cross-check tool per model on the degraded recording (figures/cross_check__<clip>__<model>.png and
results/ambiguous/cross_check__<model>.{json,md}), and writes figures/controlled_noise_audcam__<clip>.png: the aud-cam counterpart of the controlled figure
figures/controlled_noise_deg__<clip>.png, the same three models on the same degraded recording, one lane
of per-token saliency per model with the erasure test's first patch as a tick.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr

from ways.figure import from_attribution
from ways.latent import token_cam, token_cam_summary
from ways.metrics import token_mass, token_summary
from ways.provenance import write as write_manifest
from ways.tools import aud_cam, cross_check, sound_token
from ways.tools.cross_check import table as cross_table
from ways.tools.sound_token import backend

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache/controlled"
OUT = HERE / "results/ambiguous"
FIG = HERE.parent / "figures"
CLIP = "2094-142345-0024"
MODELS = {"whisper-tiny": ("hf-whisper", "openai/whisper-tiny"), "whisper-base": ("hf-whisper", "openai/whisper-base"),
          "moonshine-base": ("hf-moonshine", "UsefulSensors/moonshine-base")}
PROV = HERE / "results/controlled/provenance.json"
REVISIONS = json.load(open(PROV)).get("models", {}) if PROV.exists() else {}
HOP = 0.02


def _place(tc):
    """[K, L, n] token maps -> (centre s, width s) per token at the top layer; nan where the map is empty."""
    n = tc.shape[2]; t = (np.arange(n) + 0.5) * HOP
    c = tc[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(c > 0, c * np.log(c), 0.0), 1)
    centre = np.where(c.sum(1) > 0, (c * t).sum(1), np.nan)
    width = np.where(c.sum(1) > 0, np.exp(ent) * HOP, np.nan)
    return centre, width


def _order(centre):
    ok = np.isfinite(centre)
    return float(spearmanr(np.arange(len(centre))[ok], centre[ok]).statistic) if ok.sum() > 2 else float("nan")


def run():
    man = pd.read_csv(HERE / "results/controlled/manifest.csv").set_index("id")
    a, b = float(man.loc[CLIP, "win_start"]), float(man.loc[CLIP, "win_end"])
    gold = man.loc[CLIP, "text"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, methods, runs, inputs = [], [], [], {}
    for cond in ("deg", "clean"):
        wav, sr = sf.read(CACHE / f"{CLIP}_{cond}.wav", dtype="float32")
        inputs[f"{CLIP}_{cond}.wav"] = str(CACHE / f"{CLIP}_{cond}.wav")
        for name, (kind, mid) in MODELS.items():
            st = sound_token(wav, sr, backends=[(kind, mid)], window=(a, b))
            att = st.data["attributions"][0]
            m = token_mass(att)
            prof = np.where(np.isnan(m).all((1, 2))[:, None], np.nan, m.sum(1))      # [K, patches] erasure profile per token
            top = from_attribution(att)["top"]                      # the same first-ranked patch the sound-token figure draws
            erase_t = np.array([(tp[1] + 0.5) * att.patch_s if tp is not None else np.nan for tp in top])
            bands = {r["k"]: r.get("top_band") for r in token_summary(att)}
            what = f"pink noise at 0 dB over {a:.1f}-{b:.1f} s" if cond == "deg" else "the clean recording"
            stem = FIG / f"ambiguous__{CLIP}__{name}__{cond}"
            ac = aud_cam(wav, sr, backend=(kind, mid), window=(a, b), tokens=True, erase_t=erase_t, out=stem,
                         title=f"{name}, {what}: gradient saliency per token against the erasure test\ngold: '{gold}'")
            tc, tr = ac.data["token_cam"], ac.data["transcript"]
            tc_pooled, _ = token_cam(backend(kind, mid), wav, tr, pooled=True)
            np.savez_compressed(OUT / f"token_maps__{name}__{cond}.npz", grad_x_act=tc, grad_cam=tc_pooled, pieces=np.array(tr.pieces),
                                degraded=ac.data["result"].degraded, erase_t=erase_t)
            per_token = token_cam_summary(tc, ac.data["result"].degraded, tr.pieces, erase_profile=prof, erase_hop=att.patch_s)
            for r, t in zip(per_token, erase_t, strict=True):
                r["erase_t"] = float(t)
            c_ga, w_ga = _place(tc); c_gc, w_gc = _place(tc_pooled)
            for method, centre, width in (("erasure", erase_t, np.full(len(erase_t), att.patch_s)), ("grad x act", c_ga, w_ga), ("Grad-CAM", c_gc, w_gc)):
                methods.append(dict(model=name, cond=cond, method=method, rho_order=_order(centre), median_width_s=float(np.nanmedian(width)),
                                    share_in_window=float(np.nanmean((centre >= a) & (centre <= b)))))
            for r, cg, wg in zip(per_token, c_gc, w_gc, strict=True):
                rows.append(dict(model=name, cond=cond, **{k: v for k, v in r.items() if k != "cam_in_layers"}, gradcam_t=float(cg),
                                 gradcam_width_s=float(wg), erase_top_band=bands.get(r["k"]),
                                 **{f"cam_in_l{l}": v for l, v in enumerate(r["cam_in_layers"])}))
            runs.append(dict(model=name, cond=cond, text=att.text, n_tokens=len(per_token), degraded_share=ac.summary["degraded_share"],
                             per_layer=ac.summary["per_layer"], figures=dict(main=str(ac.figure), **ac.figures)))
            print(name, cond, repr(att.text), {mm["method"]: round(mm["rho_order"], 2) for mm in methods[-3:]})
    df = pd.DataFrame(rows); df.to_csv(OUT / "tokens.csv", index=False)
    pd.DataFrame(methods).to_csv(OUT / "methods.csv", index=False)
    (OUT / "clips.json").write_text(json.dumps(dict(clip=CLIP, gold=gold, window=[a, b], runs=runs), indent=1))
    write_manifest(OUT / "provenance.json", inputs={**inputs, "manifest.csv": str(HERE / "results/controlled/manifest.csv")},
                   seeds={}, models={mid: REVISIONS.get(mid) for _, mid in MODELS.values()},
                   extra=dict(clip=CLIP, window=[a, b], patch_s=0.1, n_bands=8, conditions=["deg", "clean"],
                              token_maps=["grad x act (per-frame gradient times activation)", "Grad-CAM (time-pooled weights)"]))
    return df


def summarise(df=None):
    df = pd.read_csv(OUT / "tokens.csv") if df is None else df
    me = pd.read_csv(OUT / "methods.csv")
    info = json.load(open(OUT / "clips.json"))
    a, b = info["window"]
    L = [f"**{CLIP}, pink noise at 0 dB SNR over {a:.1f}-{b:.1f} s. Gold: '{info['gold']}'**", ""]
    L += ["Three ways of placing each token in time. The tokens are in spoken order, so rho(order) near 1 means the "
          "placement follows the speech; width is the median effective width of a token's map at the top layer.", "",
          "| model | recording | method | rho(order) | median width (s) | tokens placed in the window |", "|---|---|---|---|---|---|"]
    for _, r in me.iterrows():
        L.append(f"| {r.model} | {'degraded' if r.cond == 'deg' else 'clean'} | {r.method} | {r.rho_order:.2f} | {r.median_width_s:.2f} | {r.share_in_window:.0%} |")
    L += ["", "Per token on the degraded recording (top layer). cam_t / width: where the gradient map sits and how spread it is; "
              "cam_in / erase_in: share of the gradient map / of the erasure mass on the noisy stretch; band: the frequency band "
              "(0 = lowest of 8) the erasure test ranked first.", ""]
    for name in MODELS:
        d = df[(df.model == name) & (df.cond == "deg")]
        L += [f"**{name}: '{[r for r in info['runs'] if r['model'] == name and r['cond'] == 'deg'][0]['text']}'**", "",
              "| k | token | cam_t (s) | width (s) | cam_in | erase_t (s) | erase_in | band | reading |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r.k} | {str(r.piece).strip() or '_'} | {r.cam_t:.2f} | {r.cam_width_s:.2f} | {r.cam_in:.2f} | {r.erase_t:.2f} | "
                     f"{r.erase_in:.2f} | {int(r.erase_top_band) if pd.notna(r.erase_top_band) else '-'} | {r.reading} |")
        L.append("")
    L += ["Control: tokens that appear in both transcripts of a model, and how far the erasure test and the gradient map moved them "
          "between the clean and the degraded recording. After the noise the audio is identical, so the anchors should stay.", "",
          "| model | token | erase_t clean | erase_t degraded | cam_t clean | cam_t degraded | moved (erasure, ms) | moved (gradient, ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for name in MODELS:
        dc = df[(df.model == name) & (df.cond == "clean")]; dd = df[(df.model == name) & (df.cond == "deg")]
        for piece in dd.piece:
            x, y = dc[dc.piece == piece], dd[dd.piece == piece]
            if len(x) == 1 and len(y) == 1:
                x, y = x.iloc[0], y.iloc[0]
                L.append(f"| {name} | {str(piece).strip() or '_'} | {x.erase_t:.2f} | {y.erase_t:.2f} | {x.cam_t:.2f} | {y.cam_t:.2f} | "
                         f"{1000 * (y.erase_t - x.erase_t):+.0f} | {1000 * (y.cam_t - x.cam_t):+.0f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cross_checks():
    """The cross-check tool on the degraded recording for each model: figure, JSON summary and Markdown table."""
    man = pd.read_csv(HERE / "results/controlled/manifest.csv").set_index("id")
    a, b = float(man.loc[CLIP, "win_start"]), float(man.loc[CLIP, "win_end"]); gold = man.loc[CLIP, "text"]
    wav, sr = sf.read(CACHE / f"{CLIP}_deg.wav", dtype="float32")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (kind, mid) in MODELS.items():
        res = cross_check(wav, sr, backend=(kind, mid), window=(a, b), wer_reference=gold, out=FIG / f"cross_check__{CLIP}__{name}",
                          title=f"{name} on the degraded recording (pink noise at 0 dB over {a:.1f}-{b:.1f} s): the erasure test and the "
                                f"encoder's gradient, token by token\ngold: '{gold}'")
        (OUT / f"cross_check__{name}.json").write_text(res.to_json())
        (OUT / f"cross_check__{name}.md").write_text(f"**{name}: '{res.summary['text']}'**  \n\n" + cross_table(res) + "\nTokens on the degraded "
                                                     f"audio: {res.summary['on_degraded']}\n")
        print(name, res.summary["on_degraded"], res.figure)


def three_model_figure():
    """The aud-cam counterpart of the controlled figure: spectrogram of the degraded recording, then for each of
    the three models one lane per token (per-frame gradient saliency at the top encoder layer, erasure tick)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from ways.figure import FigureStyle, LatentText, _Canvas, _rc, _save, _settle, _title_and_caption, logmel

    S, T = FigureStyle(), LatentText()
    _rc(S)
    info = json.load(open(OUT / "clips.json")); a, b = info["window"]
    wav, sr = sf.read(CACHE / f"{CLIP}_deg.wav", dtype="float32"); dur = len(wav) / sr
    maps = {m: np.load(OUT / f"token_maps__{m}__deg.npz", allow_pickle=True) for m in MODELS}
    C = _Canvas(S, dur, S.width_in, right=0.93)
    row_in = S.label_size * 1.7 / 72
    hs = [1.7] + [row_in * len(maps[m]["pieces"]) + 0.3 for m in MODELS]
    gap_in = S.font_size * 3.4 / 72
    fig = plt.figure(figsize=(S.width_in, 1.0))
    title = (f"The same clip and the same three models through aud-cam: each token's gradient saliency (grey) against the patch the "
             f"erasure test ranked first (tick)\ngold: '{info['gold']}'")
    t_in, c_in = _title_and_caption(fig, C, title, T.caption_tokens)
    H = t_in + 0.45 + sum(hs) + gap_in * len(hs) + c_in + 0.7
    fig.set_size_inches(S.width_in, H)
    gs = fig.add_gridspec(len(hs), 1, height_ratios=hs, hspace=gap_in / (sum(hs) / len(hs)), left=C.left, right=C.right,
                          top=1 - (t_in + 0.45) / H, bottom=(c_in + 0.7) / H)
    axm = fig.add_subplot(gs[0])
    C.mel(axm, logmel(wav, sr), T.y_mel, T.mel_ticks); C.bands(axm, [(a, b)]); C.tidy(axm)
    C.label(axm, f"the degraded recording: pink noise at 0 dB over {a:.1f}-{b:.1f} s (blue)")
    for i, m in enumerate(MODELS):
        z = maps[m]; tc, pieces, erase_t = z["grad_x_act"], list(z["pieces"]), z["erase_t"]
        K, n = tc.shape[0], tc.shape[2]
        ax = fig.add_subplot(gs[1 + i])
        M = tc[:, -1, :]; M = M / np.maximum(M.max(1, keepdims=True), 1e-12)
        ax.imshow(M, aspect="auto", cmap="Greys", extent=[0, n * HOP, K, 0], interpolation="nearest", vmin=0, vmax=1)
        for k, x in enumerate(erase_t):
            if np.isfinite(x):
                ax.plot([x, x], [k + 0.12, k + 0.88], color=C.col(k), lw=2.2, solid_capstyle="butt", zorder=4)
        ax.add_patch(Rectangle((a, 0), b - a, K, fill=False, edgecolor=S.degraded, lw=1.2, zorder=3))
        ax.set_yticks(np.arange(K) + 0.5); ax.set_yticklabels([f"{k} {str(pc).strip() or '_'}" for k, pc in enumerate(pieces)],
                                                                fontsize=S.label_size - 1, fontweight="bold")
        for k, lab in enumerate(ax.get_yticklabels()):
            lab.set_color(C.col(k))
        ax.set_ylabel(T.y_token, fontsize=S.label_size)
        d = pd.read_csv(OUT / "tokens.csv"); d = d[(d.model == m) & (d.cond == "deg")]
        both = int(((d.cam_in >= 0.5) & (d.erase_in >= 0.5)).sum()); er = int((d.erase_in >= 0.5).sum()); gr = int((d.cam_in >= 0.5).sum())
        C.tidy(ax, bottom=(i == len(MODELS) - 1), x_label=T.x_time)
        text = next(r["text"] for r in info["runs"] if r["model"] == m and r["cond"] == "deg")
        C.label(ax, f"{m}: '{text}'  ·  erasure puts {er} of {K} tokens on the noise, the gradient map {gr}, both {both}")
    _settle(fig, gs, True)
    out = FIG / f"controlled_noise_audcam__{CLIP}.png"
    _save(fig, out, S.dpi); plt.close(fig)
    print(out)


if __name__ == "__main__":
    summary_only = len(sys.argv) > 1 and sys.argv[1] == "--summary"
    summarise(None if summary_only else run())
    if not summary_only:
        cross_checks()
    three_model_figure()
