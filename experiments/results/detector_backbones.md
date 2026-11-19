| detector | parameters | weights | flags inside the damaged 2 s | flags outside | right class inside |
|---|---|---|---|---|---|
| whisper-small detector (AADD v3.1) | 88.6 M | AADD v3.1 release | 0.92 | 0.12 | 0.91 |
| whisper-base detector | 21.0 M | AADD recipe on LibriSpeech dev-clean | 0.83 | 0.17 | 0.77 |
| moonshine-base detector | 21.0 M | AADD recipe on LibriSpeech dev-clean | 0.74 | 0.20 | 0.58 |

| detector | whisper-tiny | whisper-base | moonshine-base | paired whisper-base − moonshine-base |
|---|---|---|---|---|
| whisper-small detector (AADD v3.1) | 0.95 [0.87, 1.04] (n=46) | 1.02 [0.91, 1.17] (n=46) | 1.04 [0.93, 1.15] (n=46) | -0.013 [-0.163, +0.127] |
| whisper-base detector | 0.97 [0.90, 1.03] (n=92) | 0.99 [0.93, 1.05] (n=92) | 0.95 [0.89, 1.01] (n=92) | +0.031 [-0.020, +0.094] |
| moonshine-base detector | 0.84 [0.78, 0.91] (n=91) | 0.94 [0.85, 1.05] (n=91) | 0.85 [0.78, 0.91] (n=91) | +0.095 [+0.018, +0.190] |
