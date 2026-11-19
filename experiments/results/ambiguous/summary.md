**2094-142345-0024, pink noise at 0 dB SNR over 0.6-2.6 s. Gold: 'munny my iron's twite told pease put it down to warm'**

Three ways of placing each token in time. The tokens are in spoken order, so rho(order) near 1 means the placement follows the speech; width is the median effective width of a token's map at the top layer.

| model | recording | method | rho(order) | median width (s) | tokens placed in the window |
|---|---|---|---|---|---|
| whisper-tiny | degraded | erasure | 0.99 | 0.10 | 38% |
| whisper-tiny | degraded | grad x act | 0.86 | 0.87 | 46% |
| whisper-tiny | degraded | Grad-CAM | -0.30 | 1.78 | 54% |
| whisper-base | degraded | erasure | 0.85 | 0.10 | 50% |
| whisper-base | degraded | grad x act | 0.89 | 0.73 | 50% |
| whisper-base | degraded | Grad-CAM | -0.35 | 1.76 | 64% |
| moonshine-base | degraded | erasure | 0.64 | 0.10 | 50% |
| moonshine-base | degraded | grad x act | 0.94 | 1.22 | 50% |
| moonshine-base | degraded | Grad-CAM | 0.46 | 2.58 | 25% |
| whisper-tiny | clean | erasure | 0.98 | 0.10 | 47% |
| whisper-tiny | clean | grad x act | 0.90 | 0.65 | 59% |
| whisper-tiny | clean | Grad-CAM | -0.25 | 1.53 | 41% |
| whisper-base | clean | erasure | 0.99 | 0.10 | 44% |
| whisper-base | clean | grad x act | 0.93 | 0.67 | 50% |
| whisper-base | clean | Grad-CAM | 0.16 | 1.21 | 62% |
| moonshine-base | clean | erasure | 0.98 | 0.10 | 44% |
| moonshine-base | clean | grad x act | 0.94 | 1.18 | 56% |
| moonshine-base | clean | Grad-CAM | -0.03 | 2.43 | 44% |

Per token on the degraded recording (top layer). cam_t / width: where the gradient map sits and how spread it is; cam_in / erase_in: share of the gradient map / of the erasure mass on the noisy stretch; band: the frequency band (0 = lowest of 8) the erasure test ranked first.

**whisper-tiny: 'Money, clients quite told. Please put it down to warm.'**

| k | token | cam_t (s) | width (s) | cam_in | erase_t (s) | erase_in | band | reading |
|---|---|---|---|---|---|---|---|---|
| 0 | Money | 1.53 | 0.76 | 0.44 | 0.75 | 0.92 | 1 | split |
| 1 | , | 1.76 | 0.77 | 0.79 | 0.95 | 0.93 | 4 | degraded |
| 2 | clients | 1.74 | 0.76 | 0.84 | 1.55 | 0.95 | 1 | degraded |
| 3 | quite | 2.13 | 1.15 | 0.76 | 2.25 | 0.93 | 1 | diffuse |
| 4 | told | 2.29 | 1.14 | 0.64 | 2.45 | 0.93 | 4 | diffuse |
| 5 | . | 2.28 | 0.87 | 0.38 | 2.75 | 0.47 | 1 | clean |
| 6 | Please | 3.25 | 0.31 | 0.08 | 3.45 | 0.09 | 5 | clean |
| 7 | put | 3.37 | 0.55 | 0.13 | 3.75 | 0.36 | 4 | clean |
| 8 | it | 2.79 | 0.95 | 0.27 | 3.95 | 0.25 | 4 | clean |
| 9 | down | 3.51 | 0.90 | 0.10 | 3.95 | 0.03 | 4 | clean |
| 10 | to | 3.30 | 1.01 | 0.18 | 4.35 | 0.13 | 1 | diffuse |
| 11 | warm | 3.77 | 0.66 | 0.11 | 4.65 | 0.11 | 3 | clean |
| 12 | . | 3.03 | 1.59 | 0.24 | 4.45 | 0.48 | 1 | diffuse |

**whisper-base: 'Money, fire and sprite told. Please put it down to warm.'**

| k | token | cam_t (s) | width (s) | cam_in | erase_t (s) | erase_in | band | reading |
|---|---|---|---|---|---|---|---|---|
| 0 | Money | 2.21 | 1.05 | 0.46 | 0.75 | 0.98 | 1 | diffuse |
| 1 | , | 1.89 | 0.47 | 0.89 | 1.85 | 0.91 | 5 | degraded |
| 2 | fire | 1.61 | 0.70 | 0.88 | 1.55 | 0.93 | 1 | degraded |
| 3 | and | 2.11 | 0.78 | 0.86 | 1.85 | 0.94 | 0 | degraded |
| 4 | sprite | 1.91 | 0.57 | 0.90 | 2.25 | 0.95 | 4 | degraded |
| 5 | told | 2.26 | 0.69 | 0.70 | 2.55 | 0.95 | 1 | degraded |
| 6 | . | 2.57 | 1.05 | 0.26 | 2.65 | 0.69 | 1 | diffuse |
| 7 | Please | 3.36 | 1.00 | 0.11 | 3.45 | 0.14 | 4 | clean |
| 8 | put | 3.48 | 0.47 | 0.09 | 3.65 | 0.42 | 4 | clean |
| 9 | it | 3.33 | 0.50 | 0.11 | 1.55 | 0.62 | 1 | split |
| 10 | down | 3.63 | 0.39 | 0.08 | 3.95 | 0.29 | 4 | clean |
| 11 | to | 3.03 | 0.87 | 0.18 | 4.35 | 0.04 | 1 | clean |
| 12 | warm | 3.65 | 0.75 | 0.22 | 4.55 | 0.07 | 3 | clean |
| 13 | . | 4.01 | 1.02 | 0.12 | 4.55 | 0.61 | 1 | diffuse |

**moonshine-base: 'Money, lions, fight cold. These put it down to warm.'**

| k | token | cam_t (s) | width (s) | cam_in | erase_t (s) | erase_in | band | reading |
|---|---|---|---|---|---|---|---|---|
| 0 | M | 1.47 | 1.55 | 0.14 | 0.85 | 0.85 | 0 | diffuse |
| 1 | oney | 1.67 | 1.37 | 0.67 | 0.95 | 0.97 | 0 | diffuse |
| 2 | , | 2.23 | 1.41 | 0.69 | 1.15 | 0.85 | 0 | diffuse |
| 3 | l | 2.06 | 1.58 | 0.69 | 1.95 | 0.87 | 0 | diffuse |
| 4 | ions | 2.37 | 1.48 | 0.73 | 1.85 | 0.94 | 4 | diffuse |
| 5 | , | 2.02 | 1.20 | 0.74 | 2.25 | 0.73 | 5 | diffuse |
| 6 | fight | 2.37 | 1.41 | 0.57 | 2.25 | 0.93 | 1 | diffuse |
| 7 | cold | 2.09 | 0.85 | 0.63 | 2.65 | 0.59 | 1 | degraded |
| 8 | . | 2.87 | 1.34 | 0.18 | 2.65 | 0.42 | 1 | diffuse |
| 9 | These | 3.71 | 1.23 | 0.05 | 3.45 | 0.09 | 5 | diffuse |
| 10 | put | 3.59 | 0.85 | 0.06 | 3.75 | 0.08 | 4 | clean |
| 11 | it | 3.57 | 0.73 | 0.04 | 3.75 | 0.44 | 4 | clean |
| 12 | down | 3.76 | 0.80 | 0.03 | 3.95 | 0.43 | 4 | clean |
| 13 | to | 3.74 | 0.83 | 0.04 | 4.35 | 0.29 | 1 | clean |
| 14 | warm | 4.01 | 1.10 | 0.09 | 4.35 | 0.07 | 1 | diffuse |
| 15 | . | 4.38 | 1.01 | 0.06 | 0.65 | 0.58 | 4 | diffuse |

Control: tokens that appear in both transcripts of a model, and how far the erasure test and the gradient map moved them between the clean and the degraded recording. After the noise the audio is identical, so the anchors should stay.

| model | token | erase_t clean | erase_t degraded | cam_t clean | cam_t degraded | moved (erasure, ms) | moved (gradient, ms) |
|---|---|---|---|---|---|---|---|
| whisper-tiny | Money | 0.95 | 0.75 | 1.33 | 1.53 | -200 | +208 |
| whisper-tiny | told | 2.65 | 2.45 | 2.51 | 2.29 | -200 | -217 |
| whisper-tiny | Please | 3.45 | 3.45 | 3.22 | 3.25 | +0 | +33 |
| whisper-tiny | put | 3.75 | 3.75 | 3.22 | 3.37 | +0 | +145 |
| whisper-tiny | it | 3.95 | 3.95 | 2.93 | 2.79 | +0 | -132 |
| whisper-tiny | down | 3.95 | 3.95 | 3.55 | 3.51 | +0 | -44 |
| whisper-tiny | to | 4.35 | 4.35 | 3.07 | 3.30 | +0 | +235 |
| whisper-tiny | warm | 4.35 | 4.65 | 3.90 | 3.77 | +300 | -128 |
| whisper-base | Money | 0.95 | 0.75 | 1.65 | 2.21 | -200 | +562 |
| whisper-base | , | 1.55 | 1.85 | 1.89 | 1.89 | +300 | +5 |
| whisper-base | Please | 3.35 | 3.45 | 3.24 | 3.36 | +100 | +116 |
| whisper-base | put | 3.65 | 3.65 | 3.30 | 3.48 | +0 | +182 |
| whisper-base | it | 3.75 | 1.55 | 3.18 | 3.33 | -2200 | +143 |
| whisper-base | down | 3.95 | 3.95 | 3.46 | 3.63 | +0 | +164 |
| whisper-base | warm | 4.35 | 4.55 | 3.70 | 3.65 | +200 | -50 |
| moonshine-base | M | 1.55 | 0.85 | 1.49 | 1.47 | -700 | -24 |
| moonshine-base | oney | 0.75 | 0.95 | 1.09 | 1.67 | +200 | +578 |
| moonshine-base | These | 3.35 | 3.45 | 3.49 | 3.71 | +100 | +217 |
| moonshine-base | put | 3.75 | 3.75 | 3.43 | 3.59 | +0 | +157 |
| moonshine-base | it | 3.75 | 3.75 | 3.43 | 3.57 | +0 | +145 |
| moonshine-base | down | 3.95 | 3.95 | 3.37 | 3.76 | +0 | +387 |
| moonshine-base | to | 4.35 | 4.35 | 3.32 | 3.74 | +0 | +422 |
| moonshine-base | warm | 4.35 | 4.35 | 4.08 | 4.01 | +0 | -63 |
