| model | clips with the same transcript | tokens | top patch unchanged | largest change of any delta (nats) | flips that differ | largest change of a clip's noise-focus ratio | s per clip, eager | s per clip, fused | speed-up |
|---|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 100 of 100 | 4034 | 4034 of 4034 | 2.1e-04 | 5 of 4259824 | 1.1e-04 | 7.4 | 5.5 | 1.36x |
| whisper-base | 100 of 100 | 3267 | 3267 of 3267 | 8.4e-05 | 1 of 3149872 | 8.3e-05 | 14.2 | 9.8 | 1.45x |
