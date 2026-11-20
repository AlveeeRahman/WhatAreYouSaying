# docs-src: the WhatAreYouSaying documentation

Sphinx sources for the documentation at https://buildforpeople.dev/docs/whatareyousaying/.

- `conf.py`: the site settings, and the hook that fixes links, images and reference anchors in included files.
- `*.md`: one file per page. Most pages include sections of the top-level README or a file in `docs/` with `{include}`, so edit those files, not the page.
- `_static/astryx.css`, `_static/glow.js`, `_static/fonts/`: the Astryx Neutral look shared with the OpenCarMind and TigerOCR docs.
- `package.py`: packs the built site and checks every link under the path it is served from.
- `build.sh`: build, pack and check. Commit `docs/site/` afterwards, since buildforpeople.dev reads it from the main branch.

```sh
pip install --require-hashes -r docs-src/requirements.txt
sh docs-src/build.sh
```
