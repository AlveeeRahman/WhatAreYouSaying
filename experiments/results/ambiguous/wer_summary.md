| model | WER clean -> degraded | words on the noisy stretch | right on it | right outside it | shared words | evidence in the noise, window words | evidence in the noise, outside words |
|---|---|---|---|---|---|---|---|
| whisper-tiny | 27% -> 45% | 4 -> 4 | 2 of 4 -> 1 of 4 | 6 of 7 -> 5 of 6 | 8 | 0.76 -> 0.89 | 0.21 -> 0.20 |
| whisper-base | 45% -> 45% | 4 -> 6 | 2 of 4 -> 2 of 6 | 5 of 8 -> 4 of 5 | 6 | 0.72 -> 0.88 | 0.23 -> 0.26 |
| moonshine-base | 36% -> 55% | 4 -> 3 | 1 of 4 -> 0 of 3 | 6 of 7 -> 5 of 7 | 7 | 0.71 -> 0.90 | 0.21 -> 0.36 |
