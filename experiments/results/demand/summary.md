**VoiceBank-DEMAND, 100 clips, three conditions.** AADD v3 flags: clean 0.00, noisy 0.69; on the alternating clips 0.77 of noisy frames and 0.42 of clean frames; 2.4 true noisy runs per clip, 1.5 AADD spans.

| model | WER clean | WER noisy | WER alternating | words placed on the right side (noisy or clean) | chance | word placed in its own segment | chance | mass in noisy audio, word spoken in noise | same, word spoken in clean |
|---|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 0.09 [0.06, 0.12] | 0.13 [0.09, 0.18] | 0.09 [0.06, 0.13] | 0.78 [0.75, 0.81] | 0.52 | 0.76 [0.72, 0.79] | 0.23 | 0.72 [0.69, 0.74] | 0.36 [0.33, 0.40] |
| whisper-base | 0.07 [0.04, 0.09] | 0.09 [0.06, 0.13] | 0.09 [0.06, 0.11] | 0.78 [0.74, 0.81] | 0.52 | 0.74 [0.70, 0.77] | 0.23 | 0.70 [0.67, 0.72] | 0.37 [0.33, 0.40] |
| moonshine-base | 0.03 [0.02, 0.05] | 0.06 [0.03, 0.09] | 0.05 [0.02, 0.08] | 0.79 [0.76, 0.82] | 0.52 | 0.75 [0.72, 0.78] | 0.23 | 0.72 [0.69, 0.74] | 0.37 [0.34, 0.40] |
