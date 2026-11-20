**whisper-tiny, all 100 test clips**

| gate | parameters | corpus WER | per-clip WER minus no gate | surprisal of the human transcript (nats per token) | surprisal minus no gate | mean gate |
|---|---|---|---|---|---|---|
| none | 0 | 0.383 |  | 0.770 |  | 1.00 |
| quantum | 32 | 0.383 | +0.000 [+0.000, +0.000] | 0.770 | -0.000 [-0.000, +0.000] | 1.00 |
| quantum, no entanglement | 32 | 0.383 | +0.000 [+0.000, +0.000] | 0.770 | +0.000 [-0.000, +0.000] | 0.99 |
| classical | 37 | 0.383 | +0.000 [+0.000, +0.000] | 0.770 | -0.000 [-0.000, +0.000] | 1.00 |

**whisper-tiny, the 58 test clips with at least half their frames flagged**

| gate | parameters | corpus WER | per-clip WER minus no gate | surprisal of the human transcript (nats per token) | surprisal minus no gate | mean gate |
|---|---|---|---|---|---|---|
| none | 0 | 0.457 |  | 0.765 |  | 1.00 |
| quantum | 32 | 0.457 | +0.000 [+0.000, +0.000] | 0.765 | -0.000 [-0.000, +0.000] | 1.00 |
| quantum, no entanglement | 32 | 0.457 | +0.000 [+0.000, +0.000] | 0.765 | -0.000 [-0.000, +0.000] | 0.99 |
| classical | 37 | 0.457 | +0.000 [+0.000, +0.000] | 0.765 | -0.000 [-0.000, +0.000] | 1.00 |

**whisper-base, all 100 test clips**

| gate | parameters | corpus WER | per-clip WER minus no gate | surprisal of the human transcript (nats per token) | surprisal minus no gate | mean gate |
|---|---|---|---|---|---|---|
| none | 0 | 0.182 |  | 0.702 |  | 1.00 |
| quantum | 32 | 0.182 | +0.000 [+0.000, +0.000] | 0.702 | -0.000 [-0.001, +0.000] | 0.97 |
| quantum, no entanglement | 32 | 0.182 | +0.001 [+0.000, +0.002] | 0.702 | -0.000 [-0.000, +0.000] | 0.98 |
| classical | 37 | 0.182 | +0.000 [+0.000, +0.000] | 0.702 | +0.000 [-0.000, +0.000] | 1.00 |

**whisper-base, the 58 test clips with at least half their frames flagged**

| gate | parameters | corpus WER | per-clip WER minus no gate | surprisal of the human transcript (nats per token) | surprisal minus no gate | mean gate |
|---|---|---|---|---|---|---|
| none | 0 | 0.184 |  | 0.693 |  | 1.00 |
| quantum | 32 | 0.184 | +0.000 [+0.000, +0.000] | 0.693 | +0.000 [-0.001, +0.001] | 0.97 |
| quantum, no entanglement | 32 | 0.184 | +0.000 [+0.000, +0.000] | 0.693 | +0.000 [-0.000, +0.000] | 0.98 |
| classical | 37 | 0.184 | +0.000 [+0.000, +0.000] | 0.693 | -0.000 [-0.000, +0.000] | 1.00 |
