# Reproducibility audit against Vangala et al. (2026)

This audit takes Vangala, Adibifar, Gehani and Malik, "AI-Generated Code Is Not Reproducible (Yet):
An Empirical Study of Dependency Gaps in LLM-Based Coding Agents" (arXiv:2512.22387 v3, 2026) as a
checklist and runs WAYS through it. Every row below was checked by running a command or reading the
file named. Audit date: 2026-11-18, starting from commit `127abb5`.

## 1. What the paper says

**Executable reliability (p. 1, p. 3, Eq. 2).** The fraction of prompts for which an agent's code
"executes without any manual intervention". `exec` returns success only when the code runs using
exclusively the dependencies the LLM specified, in a standardized clean environment that holds only
the default operating system packages. Any debugging or dependency addition counts as a failure.

**Clean-environment protocol (p. 4).** AWS EC2 t2.large (4 vCPU, 16 GB), Ubuntu 22.04, reset
between projects to a baseline of exactly 91 packages (71 system, 20 user, including SciUnit
v0.4.post135), checked after each test (Eqs. 9 and 10). Runtime dependencies were captured with
SciUnit for Python, `npm list --all` for JavaScript and `mvn dependency:tree` for Java (Eqs. 11 to 13).
Algorithm 1 (p. 4) is used only for post-failure analysis. It installs the claimed set, runs the
code, adds the missing package on an ImportError, applies a minimal fix on a SyntaxError or logic
error, and marks anything else (for example conflicting version requirements) as unfixable. It loops
up to MAX_ITERATIONS, which the paper does not give as a number. In practice it took 2 to 3
iterations. Successes reported anywhere in the paper count only out-of-the-box runs, never projects
fixed by Algorithm 1.

**Three dependency layers (p. 3).**

* Claimed, D_c = {d : d in parse(D_i)} (Eq. 3): what the requirements file lists.
* Working, D_w = D_c ∪ D_m (Eq. 4): D_c plus the missing packages D_m found through ImportErrors.
* Runtime, D_r = ⋃ over d in D_w of {d} ∪ transitive(d) (Eq. 5): everything that loads.
* Completeness gap, Δ_completeness = |D_w| − |D_c| (Eq. 6).
* Runtime multiplier, ρ_runtime = |D_r| / |D_c| (Eq. 7).

**Failure causes and numbers.**

* Overall 205 of 300 projects (68.3 %) ran out of the box, 95 (31.7 %) failed (p. 1, p. 4, Table 1
  p. 5). Table 1 also has a "Partial" column (14 projects) for code that runs but needs external
  services such as databases or APIs.
* Missing dependencies and ImportError: 10.5 % of the failures, 10 of the 95 (p. 1, p. 6).
* Fundamental code errors were the majority: malformed syntax, incorrect file paths, uninitialized
  variables and structural issues (p. 1). The paper names them but gives no per-cause percentage.
* Language-specific issues (p. 5): Python 89.2 %, JavaScript 61.9 %, Java 44.0 % (Eq. 14, Table 2).
  Python does well because of flat requirements files and the pip resolver, JavaScript suffers from
  nested npm dependencies, and Java from pom.xml configuration, deep transitive graphs, dependency
  scopes and version conflicts.
* Completeness gap (p. 6, Eq. 16): P(Δ = 0) = 0.87, P(Δ = 1) = 0.08, P(Δ = 2) = 0.03,
  P(Δ ≥ 3) = 0.02. So 13 % of projects needed 1 to 3 packages added by hand. Common omissions were
  lxml, python-dotenv and bcrypt (Python), body-parser, ws and dotenv (JavaScript), and JUnit and SLF4J
  (Java).
* Runtime multiplier: 13.5x on average (abstract). 3 declared packages against 37 loaded (p. 1).
  Java 9.5x (2.2 declared, 20.8 loaded), JavaScript 1.0x, Python 2.5x (Fig. 5, p. 6).
* Cost: about 15 minutes of manual debugging per failed project (p. 1, p. 5).

**Prompt template (Figure 1, p. 3).** "Provide EVERYTHING needed for reproduction in [Environment]:
1. Complete working [Language] code 2. Complete requirements.txt/package.json/pom.xml with ALL
dependencies and versions 3. Brief usage instructions. Make it 100% reproducible in a clean
[environment]." For WAYS the same three deliverables are the code, `requirements.txt` with
`pyproject.toml`, and the README and `docs/reproducibility.md`.

## 2. Each failure cause, applied to WAYS

| cause (paper) | WAYS before | evidence | status now |
|---|---|---|---|
| Out-of-the-box run in a clean env (Eq. 2) | not exposed for the CPU path | `bash scripts/reproduce_clean_env.sh` before any edit: `12 passed, 2 deselected`, "clean-environment run passed", exit 0 | still passes, and the check is now wider (below) |
| Clean-env check covers only part of the code | exposed | the old script imported only `ways.*`. `transformers`, `aadd`, `librosa`, `pandas` and `openai` were never imported, so the recorded D_r had 16 entries and no `aadd` or `transformers` | fixed: script now runs `pip check`, `ways explain --help` and imports every third-party module the code uses |
| Missing dependency, ImportError (10.5 %) | exposed, figure scripts only | `experiments/gif_erase_sweep.py:11` imports `PIL` and `docs/images/render.py:9` imports `playwright`. Neither was in `requirements.txt` or `pyproject.toml`. Pillow arrived as a transitive dependency of matplotlib, at 13.0.0 in the clean venv against 12.2.0 in the environment that drew the GIF. Playwright was absent, so `render.py` would fail with ImportError | fixed: `pillow==12.2.0` and `playwright==1.65.2` in `requirements.txt`, a `figures` extra in `pyproject.toml`, the Chromium step documented |
| Claimed but never imported | present, harmless | `torchaudio` is pinned but imported nowhere, neither by WAYS nor by AADD. `scipy` is never imported by WAYS but is a requirement of AADD. `pytest` is used as a tool | left as is (see section 4) |
| Direct-URL dependency (AADD not on PyPI) | not exposed | `packaging.requirements.Requirement` parses the `aadd @ git+https://...@c6a2debc...` line. `pip install --dry-run .` resolves it ("Using cached aadd-3.1.0-py3-none-any.whl, Would install ways-0.1.0"). `pip check` in the clean venv: "No broken requirements found" | no change needed |
| Incorrect file paths | exposed | an absolute path to the author's own venv was the default interpreter in `experiments/run_all.sh:5`, `run_controlled.sh:5`, `run_detectors.sh:5` and `run_backbones.sh:4`. `tests/test_openai_compat.py:17` hard-coded an absolute clip path. The data path was a separate default in six scripts (from `argv[1]` in five, from `argv[4]` in `run_models.py`), and `gif_erase_sweep.py:31` hard-coded it with no override. A missing repository gave a bare pandas FileNotFoundError | fixed: one resolver, `experiments/common.py:data_root`, with order argument, then `WAYS_DATA`, then `../ways-data`, and a message describing the folder layout. Scripts use `$PY` (default `python`) and the test reads `WAYS_DATA` |
| Script that never terminates | exposed | `run_detectors.sh:7` waited forever for `experiments/cache/librispeech/download_dev.log` to contain DONE. No file in the repository writes that log, so on a fresh clone `run_backbones.sh` hangs | fixed: checks for `LibriSpeech/dev-clean` and exits with the download URL |
| Failures hidden by the run scripts | exposed | the scripts used `set -u` and piped Python through `grep -v`, so a crashed model run still exited 0 | fixed: `set -euo pipefail` |
| Malformed syntax | not exposed | `py_compile` on every tracked `.py` file passes. `ast.parse(..., feature_version=(3,10))` and `(3,11)` raise nothing. A tokenizer scan found no 3.12-only f-string nesting. `ruff --select F821,F823` finds no undefined name | none needed |
| Uninitialized variables | not exposed | same ruff run: no F821 or F823. It does report 9 unused imports and one unused local (`ways/erasure.py:134`), which are harmless | none needed |
| Structural issues, order of steps | partly exposed | `compare.py`, `vox_mixed_figures.py` and `detector_backbones.py` read `npz` maps that `.gitignore` excludes, so they only work after `run_all.sh`. The documented order is right. LibriSpeech was named without a URL or directory layout. The README gave no prerequisites for a first `ways explain` (weights download, AADD checkpoint) | fixed in the docs: a prerequisites paragraph, download commands, the missing `gif_erase_sweep.py` and `render.py` steps |
| Python version (language-specific) | exposed | `pyproject.toml:10` claimed `>=3.10`. The pinned set needs more: numpy 2.5.2 and scipy 1.18.0 declare `Requires-Python >=3.12`, and pandas 3.0.3, matplotlib 3.11.2 and pillow need `>=3.11`. The code itself parses under 3.10 | fixed: `requires-python = ">=3.12"`, the only version tested (3.12.3) |
| Build backend deprecation | exposed, dated | the clean-env log warns that `license = {text = "MIT"}` stops building after 2027-02-18 | fixed: `license = "MIT"`, `setuptools>=77` |
| Version pinning of packages | partly exposed | `requirements.txt` pins all 14 direct packages exactly. Transitive packages are not pinned and drift (pillow 12.2.0 against 13.0.0, tqdm 4.70.0 against 4.70.1) | direct pins kept. A lock file is a decision (section 4) |
| Model weights from "latest" | exposed | `WhisperBackend` and `MoonshineBackend` accept `revision=` but `run_models.py:45`, `example_clip.py`, `controlled.py` and `train_detectors.py` never passed it. `provenance.py` then recorded the newest cached snapshot, not necessarily the loaded one | fixed: `REVISIONS` in `experiments/common.py`, using the commits from `experiments/results/*/provenance.json`, passed as `revision=` in every experiment. `ways explain --revision`. `manifest()` records the pinned revision. Loading each model offline at its pin worked on this machine |
| AADD checkpoints | exposed, outside WAYS | AADD downloads from `huggingface.co/gertelrina/aadd/resolve/main/...` with no revision and no hash | documented sha256 of the three files used. Not pinnable without changing AADD |
| Seeds and determinism | not exposed | sample, bootstrap, controlled set and detector training all use seed 8001 through `default_rng` or `torch.manual_seed`. No bare `np.random.*` or `random.*` call in any tracked file. `rough.js` uses a seeded wobble. Re-drawing `default_rng(8001).choice(1500, 100)` on the hashed `test.csv` gives exactly `experiments/sample100.csv` | none needed |
| External services ("Partial", Table 1) | exposed, stated | the `openai-compat` backend needs a live vLLM or llama.cpp server. Only the mock server was tested, and the README says so | none needed |
| Runtime provenance (D_r, SciUnit role) | partly exposed | the three `provenance.json` files have `code_commit: null` because the runs predate the first commit. Their D_r (89 packages, multiplier 6.3x) was measured in the development environment, which holds more than `requirements.txt` installs | documented in `docs/reproducibility.md`. Data files left untouched |
| Docs misquoting the paper | exposed | `docs/reproducibility.md:7-9` said the main causes were unlisted dependencies. The paper says ImportErrors were 10.5 % of failures and most were code errors | fixed |
| Usage instructions (Fig. 1, item 3) | partly exposed | every documented CLI flag exists in `ways/cli.py` and every referenced file exists. Missing were `WAYS_DATA`, offline mode on a first run, the LibriSpeech URLs and the Chromium install | fixed in `docs/reproducibility.md` "Re-running" |

Tests: `.venv/bin/python -m pytest tests -q -m "not gpu and not slow"` gives `12 passed, 2
deselected` before and after the edits. The 2 deselected tests are the GPU black-box tests, which
need the example clip from the data folder and cached whisper-tiny.

## 3. Edits

* `experiments/common.py` (new): `data_root()` and `REVISIONS`, with a docstring.
* `experiments/sample_test100.py`, `run_models.py`, `run_aadd_variants.py`, `example_clip.py`,
  `vox_mixed_figures.py`, `detector_backbones.py`, `gif_erase_sweep.py`: the data path now comes
  from `data_root()`. `gif_erase_sweep.py` takes the path as an argument.
* `run_models.py`, `example_clip.py`, `controlled.py`: `load_backend(..., revision=REVISIONS.get(id))`.
  `run_models.py` passes the pinned revision to the manifest.
* `train_detectors.py`: the Moonshine encoder loads at its pinned revision.
* `experiments/run_all.sh`, `run_controlled.sh`, `run_backbones.sh`, `run_detectors.sh`: no absolute
  interpreter path, `set -euo pipefail`, offline mode can be overridden. `run_detectors.sh` checks for
  dev-clean instead of waiting on a log file.
* `ways/cli.py`: `--revision`. `ways/provenance.py`: `manifest(models=...)` accepts `{id: revision}`.
* `tests/test_openai_compat.py`: the clip path comes from `WAYS_DATA` or the sibling folder.
* `requirements.txt`: `pillow==12.2.0`, `playwright==1.65.2`, with a comment.
* `pyproject.toml`: `requires-python = ">=3.12"`, SPDX licence string, `setuptools>=77`, `figures` extra.
* `scripts/reproduce_clean_env.sh`: writes its own log, runs `pip check`, `ways explain --help` and an
  import check of every third-party module.
* `README.md`: prerequisites before the first command, model pins, new clean-env numbers, links to
  this audit, `common.py` in the layout table.
* `docs/reproducibility.md`: corrected summary of the paper, pinned weights and AADD hashes, the
  provenance caveat, the wider clean-env test, prerequisites and download commands in "Re-running".
* `provenance/clean_env_run.json` and `.log`: rewritten by the final clean-env run.

Later (2026-11-19, after the VoiceBank-DEMAND set was added): the README claimed that every run
writes `provenance.json`, but only `run_models.py` did. `controlled.py`, `latent_run.py` and
`demand.py` now write one at the end of each model pass (the finished passes were re-entered, which
skips the done clips and writes the manifest, so their `code_commit` is the commit of that re-entry,
not of the original run). The clean-environment test was re-run on the final tree.

## 4. Not fixed, and why

* **AADD weights are fetched from `main`.** The URLs live inside AADD (`aadd/model.py`). Pinning
  them needs a change upstream, or a pre-download step in WAYS that checks the sha256 values listed in
  `docs/reproducibility.md`. I did not add that step.
* **The whisper-base detector backbone has no revision pin.** It is built by AADD's
  `WhisperFrameModel(backbone="openai/whisper-base")`, which takes no revision. At inference the
  trained checkpoint overwrites the weights, so only retraining depends on it.
* **Transitive packages are unpinned.** `requirements.txt` pins the direct layer, which is what the
  paper asks for. Pinning the transitive layer needs a lock file (`pip freeze` from the clean venv).
  That is the authors' call.
* **`torchaudio` is claimed but unused.** Removing it shrinks the install and changes nothing I
  could find, but the results were produced with it present, so I left the decision to the authors.
* **The GPU runs were not re-executed** (`run_all.sh`, `run_controlled.sh`, `run_backbones.sh`), as
  agreed. Executable reliability is shown here for the CPU path and the import layer only.
* **`code_commit` is null in the three 100-clip `provenance.json` files**, and their D_r comes from a
  fuller environment. These are data files, so they were not edited.
* **The hand-drawn figures load their fonts from Google Fonts** at render time, although the same
  TTF files are in `docs/images/fonts/`. Switching to `@font-face` with local files could shift the
  rendering, so it is noted in the docs rather than changed.
* **HEAD moved during the audit.** Commit `c5383a5` (detector figure) was made in parallel. It adds
  stdlib-only scripts and no new dependency.

## 5. Clean-environment run, before and after

| | before (commit `127abb5`) | after (this audit) |
|---|---|---|
| result | passed, exit 0 | passed, exit 0 |
| `pip check` | not run | "No broken requirements found" |
| tests | 12 passed, 2 deselected | 12 passed, 2 deselected |
| `ways explain --help` and import check | not run | passed |
| claimed packages (size of D_c) | 14 | 16 |
| distributions loaded (excluding `ways`) | 15 | 59 |
| loaded but not claimed | 11 | 45 |
| multiplier as the script prints it | 1.1x | 3.8x |

The after column comes from `provenance/clean_env_run.json`. The multiplier rose because the check
now loads the whole stack (transformers, AADD, librosa, openai), not because anything got worse. The
committed JSON before the audit also listed `ways` itself under "not claimed", which the script
already filters out. Re-running it unchanged removed that entry, so that copy had been produced by
an older version of the script.
