# WhatAreYouSaying documentation

WhatAreYouSaying (WAYS) is an attribution tool for speech-to-text models. For every token a model wrote, it reports which 100 ms of audio in which frequency band the token depended on, and whether that audio was clean or degraded. It runs on local Whisper and Moonshine weights and on any OpenAI-compatible speech server such as vLLM or llama.cpp. It needs no training and no labels, and the cost is one forward pass per erased patch.

```{image} https://raw.githubusercontent.com/AlveeeRahman/WhatAreYouSaying/main/docs/images/erase_sweep.gif
:alt: One patch at a time is erased from a clip's spectrogram, and the bars show how much less sure whisper-base became of every word
```

The method combines representation erasure by Li, Monroe and Jurafsky, which measures what a network uses by deleting parts of its input, with AADD by Arina Gertel, which labels what is wrong with a recording and where, every 20 ms. The code is MIT licensed.

## Start here

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Install and use
:link: install-and-use
:link-type: doc
Set up Python 3.12, run your first clip, and call WAYS from Python.
:::

:::{grid-item-card} Command line
:link: cli
:link-type: doc
Every `ways` command and option, read from the source.
:::

:::{grid-item-card} How it works
:link: how-it-works
:link-type: doc
The erasure grid, the three scores per token, and the noise-focus ratio.
:::

:::{grid-item-card} Tools
:link: tools
:link-type: doc
`sound-token`, `aud-cam` and `cross-check`, and how to add a tool.
:::

:::{grid-item-card} Backends
:link: backends
:link-type: doc
Local Whisper and Moonshine weights, vLLM and llama.cpp servers.
:::

:::{grid-item-card} Results
:link: results
:link-type: doc
100 VoxCeleb clips, controlled damage, real noise and the latent space.
:::

::::

(what-it-answers)=
## What it answers

```{include} ../README.md
:start-after: "## What it answers"
:end-before: "## How it works"
```

```{toctree}
:hidden:

install-and-use
Command line <cli>
how-it-works
Tools <tools>
Backends <backends>
Figure <figure>
results
Reproducibility <reproducibility-guide>
Audit <reproducibility-audit>
Limitations <limitations>
help-wanted
references
Layout and credits <layout>
changelog
```
