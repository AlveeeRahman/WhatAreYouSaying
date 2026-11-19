"""Render the hand-drawn figure sources in docs/images/src to PNG at 3x with the Playwright Chromium
(user-local, no system install), the way OpenCarMind's docs/images/render.sh does with Chrome.
Run from the repository root:  python docs/images/render.py [name ...]
Each source reports missing fonts and text that runs past the page in data attributes on <body>;
the script prints both so a bad render never passes silently."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
FIGS = {  # name: (width, height, output)
    "workflow_method": (1200, 1350, HERE / "workflow_method.png"),
    "workflow_experiments": (1200, 1330, HERE / "workflow_experiments.png"),
    "workflow_reproducibility": (1200, 770, HERE / "workflow_reproducibility.png"),
    "detector_backbones": (1200, 830, HERE.parents[1] / "figures/detector_backbones.png"),   # built by src/make_detector_figure.py
    "demand_tracking": (1200, 830, HERE.parents[1] / "figures/demand_tracking.png"),         # built by src/make_demand_figure.py
}
SCALE = 3


def shot(page, name, w, h, out, step=None):
    url = (SRC / f"{name}.html").as_uri() + (f"?step={step}" if step is not None else "")
    page.set_viewport_size({"width": w, "height": h})
    page.goto(url)
    page.wait_for_function("document.body.dataset.fonts !== undefined", timeout=30000)
    fonts, over = page.evaluate("[document.body.dataset.fonts, document.body.dataset.overflow]")
    page.screenshot(path=str(out), full_page=False)
    print(f"rendered {out.name}  fonts={fonts}  overflow={over or 'none'}")
    return fonts, over


def main(names):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=SCALE)
        for n in names or FIGS:
            w, h, out = FIGS[n]
            shot(page, n, w, h, out)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
