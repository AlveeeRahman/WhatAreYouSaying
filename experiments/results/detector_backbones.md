| detector | parameters | weights | flags inside the damaged 2 s | flags outside | right class inside |
|---|---|---|---|---|---|
| whisper-small detector (AADD v3.1) | 88.6 M | AADD v3.1 release | 0.92 | 0.12 | 0.91 |
| whisper-base detector | 21.0 M | AADD recipe on LibriSpeech dev-clean | 0.83 | 0.17 | 0.77 |
| moonshine-base detector | 21.0 M | AADD recipe on LibriSpeech dev-clean | 0.74 | 0.20 | 0.58 |

| detector | whisper-tiny | whisper-base | moonshine-base | paired whisper-base − moonshine-base |
|---|---|---|---|---|
| whisper-small detector (AADD v3.1) | 0.93 [0.83, 1.03] (n=22) | 1.04 [0.92, 1.18] (n=22) | 1.04 [0.93, 1.15] (n=22) | +0.004 [-0.088, +0.097] |
| whisper-base detector | 1.01 [0.96, 1.07] (n=65) | 1.03 [0.99, 1.07] (n=65) | 1.02 [0.97, 1.07] (n=65) | +0.007 [-0.023, +0.036] |
| moonshine-base detector | 0.84 [0.78, 0.91] (n=67) | 0.90 [0.84, 0.96] (n=67) | 0.86 [0.79, 0.92] (n=67) | +0.043 [+0.002, +0.086] |
