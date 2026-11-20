"""Command line.

  ways explain clip.wav --backend hf-whisper --model openai/whisper-base --out out/
  ways explain clip.wav --backend hf-moonshine --model UsefulSensors/moonshine-base --out out/
  ways explain clip.wav --backend openai-compat --base-url http://localhost:8000/v1 --model openai/whisper-large-v3 --out out/
  ways tools                                 # the registered tools; --schema prints their JSON schemas
  ways sound-token clip.wav --backend hf-whisper=openai/whisper-base --backend hf-moonshine=UsefulSensors/moonshine-base --aadd --out out/clip
  ways aud-cam clip.wav --backend hf-whisper=openai/whisper-base --aadd --granular --out out/clip
  ways quality clip.wav                      # AADD v3 degradation timeline, 20 ms
  ways latent clip.wav --backend hf-whisper --model openai/whisper-base --aadd --figure --out out/
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _load(path):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def cmd_explain(a):
    from . import explain, load_backend
    from .metrics import clip_summary, token_summary

    wav = _load(a.audio)
    kw = {}
    if a.backend == "openai-compat":
        kw = dict(base_url=a.base_url, route=a.route)
    elif a.revision:
        kw = dict(revision=a.revision)
    be = load_backend(a.backend, a.model, **kw)
    att = explain(be, wav, patch_s=a.patch, n_bands=a.bands, batch=a.batch, log=print if a.verbose else None)
    probs = None
    if a.aadd:
        from .quality import Quality

        probs = Quality(a.aadd_model).frames(wav)
    rows = token_summary(att, probs)
    summ = clip_summary(att, rows, probs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    att.save(out / f"{stem}.npz")
    (out / f"{stem}.json").write_text(json.dumps(dict(summary=summ, tokens=rows), indent=1, default=float))
    if probs is not None:
        np.save(out / f"{stem}_aadd.npy", probs)
    if a.figure:
        from .plots import clip_figure

        clip_figure(att, probs, wav, out / f"{stem}.png", title=f"{stem}  ·  {att.model}")
    print(json.dumps(summ, indent=1, default=float))
    print("transcript:", att.text)
    for r in rows:
        print(f"  {r['k']:3d} {r['piece']!r:16} top {r.get('top_centre_s', float('nan')):5.2f}s band {r.get('top_band', -1)} "
              f"share {r.get('top_share', float('nan')):.2f} focus {r.get('focus_ratio', float('nan')):.2f}")


def cmd_latent(a):
    """Per-layer Grad-CAM and spill-over for one clip (white-box backends only)."""
    from . import load_backend
    from .latent import latent_view

    if not a.aadd and not a.window:
        raise SystemExit("ways latent needs --aadd or --window: without a degraded stretch there is nothing to erase")
    wav = _load(a.audio)
    be = load_backend(a.backend, a.model, **(dict(revision=a.revision) if a.revision else {}))
    probs = None
    if a.aadd:
        from .quality import Quality

        probs = Quality(a.aadd_model).frames(wav)
    window = tuple(a.window) if a.window else None
    r = latent_view(be, wav, probs=probs, window=window)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    np.savez_compressed(out / f"{stem}_latent.npz", cam=r.cam, shift=r.shift, degraded=r.degraded, loudness_db=r.loudness_db)
    (out / f"{stem}_latent.json").write_text(json.dumps(r.to_dict(), indent=1, default=float))
    if a.figure:
        from .figure import latent_figure
        from .quality import CLASSES

        latent_figure(wav, 16000, quality=probs, quality_classes=CLASSES if probs is not None and probs.shape[1] == 12 else None,
                      cam=r.cam, shift=r.shift, degraded=r.degraded, loudness_db=r.loudness_db, summary=r.summary,
                      model_name=r.model, title=f"{stem}  ·  {r.model}", out=out / f"{stem}_latent.png")
    print(f"{r.model}: {r.layers} layers, {r.n_frames} frames, degraded share {r.degraded.mean():.0%}")
    for s in r.summary:
        print(f"  layer {s['layer']:2d}  spill-over outside {s['shift_out']:.3f}  shift inside {s['shift_in']:.3f}  "
              f"CAM focus {s['cam_focus']:.2f}  rho(loud) {s['rho_loud']:+.2f}")


def cmd_tool(a):
    """Any registered tool: `ways <tool> clip.wav ...` through ways.tools.call."""
    from .tools import call, get

    tool = get(a.cmd)
    given = bool(a.backend)
    a.backend = a.backend or ["hf-whisper=openai/whisper-base"]
    specs = [b.split("=", 1) if "=" in b else [b, None] for b in a.backend]
    kw = dict(audio=a.audio, aadd=a.aadd, out=a.out, title=a.title)
    if a.window:
        kw["window"] = a.window
    props = tool.parameters.get("properties", {})
    if "backends" in props:
        if given or "default" not in props["backends"]:
            kw["backends"] = specs
    elif "backend" in props:
        kw["backend"] = specs[0]
    if "wer_reference" in props and a.reference:
        kw["wer_reference"] = a.reference
    if "granular" in props:
        kw["granular"] = a.granular
    if "tokens" in props:
        kw["tokens"] = a.tokens
    for key in ("title", "window"):
        if key not in props:
            kw.pop(key, None)
    print(call(a.cmd, **kw).to_json())


def cmd_tools(a):
    """List the registered tools, or print their function-calling schemas."""
    from .tools import schema_json, tools

    if a.schema:
        print(schema_json())
        return
    for t in tools():
        box = "white-box backends" if t.needs_white_box else "any backend"
        print(f"{t.name:14} {box:20} {t.summary}")


def cmd_quality(a):
    from .quality import Quality, intervals

    wav = _load(a.audio)
    probs = Quality(a.aadd_model).frames(wav)
    fl = (probs >= 0.5).any(1)
    print(f"{len(probs)} frames of 20 ms, {fl.mean():.0%} flagged at p >= 0.5")
    for name, s, e, pk in intervals(probs):
        print(f"  {name:13} {s:6.2f}-{e:6.2f} s  peak {pk:.2f}")
    if a.out:
        np.save(a.out, probs)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ways", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explain", help="per-token erasure attribution for one clip")
    e.add_argument("audio")
    e.add_argument("--backend", default="hf-whisper", choices=["hf-whisper", "hf-moonshine", "openai-compat"])
    e.add_argument("--model", default=None)
    e.add_argument("--revision", default=None, help="Hugging Face commit of the model (hf-* backends); default: latest")
    e.add_argument("--base-url", default="http://localhost:8000/v1")
    e.add_argument("--route", default="transcriptions", choices=["transcriptions", "chat"])
    e.add_argument("--patch", type=float, default=0.1, help="patch length in seconds (multiple of 0.01)")
    e.add_argument("--bands", type=int, default=8)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--aadd", action="store_true", help="also run AADD and report the noise-focus ratio")
    e.add_argument("--aadd-model", default="v3")
    e.add_argument("--figure", action="store_true")
    e.add_argument("--out", default="ways_out")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(fn=cmd_explain)
    l = sub.add_parser("latent", help="per-layer Grad-CAM and spill-over under erasure of the degraded audio (white-box backends)")
    l.add_argument("audio")
    l.add_argument("--backend", default="hf-whisper", choices=["hf-whisper", "hf-moonshine"])
    l.add_argument("--model", default=None)
    l.add_argument("--revision", default=None)
    l.add_argument("--aadd", action="store_true", help="erase the stretches AADD flags; without it give --window")
    l.add_argument("--aadd-model", default="v3")
    l.add_argument("--window", type=float, nargs=2, metavar=("START_S", "END_S"), help="erase this stretch instead of AADD's flags")
    l.add_argument("--figure", action="store_true")
    l.add_argument("--out", default="ways_out")
    l.set_defaults(fn=cmd_latent)
    from .tools import tools as registered_tools
    for tool in registered_tools():
        t = sub.add_parser(tool.name, help=tool.summary.split(". ")[0].rstrip("."))
        t.add_argument("audio")
        t.add_argument("--backend", action="append", default=None, metavar="KIND=MODEL",
                       help="hf-whisper=openai/whisper-base (repeat for several where the tool takes a list); hf-moonshine=UsefulSensors/moonshine-base")
        t.add_argument("--aadd", action="store_true", help="run AADD v3 to find degraded audio")
        t.add_argument("--window", type=float, nargs=2, metavar=("START_S", "END_S"), help="a known damaged span instead of AADD's flags")
        t.add_argument("--reference", default=None, help="human transcript, to report WER (tools that take wer_reference)")
        t.add_argument("--granular", action="store_true", help="also write each lane as its own figure (tools that take granular)")
        t.add_argument("--tokens", action="store_true", help="also compute Grad-CAM per token (tools that take tokens)")
        t.add_argument("--title", default=None)
        t.add_argument("--out", default="ways_out/clip")
        t.set_defaults(fn=cmd_tool)
    ls = sub.add_parser("tools", help="list the registered tools, or --schema for their function-calling schemas")
    ls.add_argument("--schema", action="store_true")
    ls.set_defaults(fn=cmd_tools)
    q = sub.add_parser("quality", help="AADD degradation timeline for one clip")
    q.add_argument("audio")
    q.add_argument("--aadd-model", default="v3")
    q.add_argument("--out", default=None)
    q.set_defaults(fn=cmd_quality)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
