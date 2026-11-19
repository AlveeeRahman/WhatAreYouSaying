"""The example figures of the tools in the README, on one partly degraded VoxCeleb clip.

aud-cam on whisper-base and moonshine-base gives figures/latent__<model>__<clip>.png. The granular
lanes of the whisper-base run give figures/tool_aud_cam__vox_mixed__whisper-base{,__cam,__layers,__spillover}.png.
sound-token on the three models gives figures/tool_sound_token__vox_mixed.png. The degraded stretch
comes from the AADD scan in the data folder. Usage: python experiments/tool_examples.py [data folder]"""
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from common import AADD_CSV, AUDIO_DIR, REVISIONS, data_root

from ways.quality import load_scan_csv
from ways.tools import aud_cam, sound_token

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
DATA = data_root(sys.argv[1] if len(sys.argv) > 1 else None, need=(AUDIO_DIR, AADD_CSV))
CLIP = "id10274_89UnsytX6Y8_00006"
MODELS = {"whisper-tiny": ("hf-whisper", "openai/whisper-tiny"), "whisper-base": ("hf-whisper", "openai/whisper-base"),
          "moonshine-base": ("hf-moonshine", "UsefulSensors/moonshine-base")}


def keep(src, dst):
    """Copy a figure and its SVG twin, when the tool wrote one, to its name in figures/."""
    shutil.copyfile(src, dst)
    if Path(src).with_suffix(".svg").exists() and dst.with_suffix(".svg").exists():
        shutil.copyfile(Path(src).with_suffix(".svg"), dst.with_suffix(".svg"))
    print("wrote", dst.name)


def main():
    wav, sr = sf.read(DATA / AUDIO_DIR / f"{CLIP}.wav", dtype="float32")
    probs = load_scan_csv(DATA / AADD_CSV)[CLIP]
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("whisper-base", "moonshine-base"):
            kind, mid = MODELS[name]
            res = aud_cam(wav, sr, backend=(kind, mid), quality=probs, granular=name == "whisper-base", revision=REVISIONS[mid],
                          out=Path(tmp) / name, title=f"{CLIP}  ·  {name}")
            keep(res.figure, FIG / f"latent__{name}__{CLIP}.png")
            if name == "whisper-base":
                keep(res.figure, FIG / "tool_aud_cam__vox_mixed__whisper-base.png")
                for lane in ("cam", "layers", "spillover"):
                    keep(res.figures[lane], FIG / f"tool_aud_cam__vox_mixed__whisper-base__{lane}.png")
        res = sound_token(wav, sr, backends=list(MODELS.values()), quality=probs, revision=REVISIONS, out=Path(tmp) / "st", title=f"{CLIP}: sound-token")
        keep(res.figure, FIG / "tool_sound_token__vox_mixed.png")


if __name__ == "__main__":
    main()
