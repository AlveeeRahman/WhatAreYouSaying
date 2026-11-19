"""Which audio does a Whisper token timestamp point at? A check of the indexing in ways/backends/hf_whisper.py.

Two clean LibriSpeech utterances A and B are joined with one second of silence between them, so the
time where B starts is known. If timestamp k is the start of token k, the last word of A starts
before A ends and the first token of B starts at B's own speech onset. If the timestamps were one
token late, the last word of A would carry the time of the full stop after it or of B. The full
stop that ends A has no sound of its own, so it is reported separately. A pair is used only when the
joined transcript starts with exactly the tokens the model writes for A alone, so the token where B
starts is known too. Speech onset is the first 20 ms frame within 30 dB of the utterance's loudest.

Needs the controlled set (experiments/controlled.py prepare). Writes experiments/results/checks/timestamps.{json,md}.
Usage: python experiments/check_timestamps.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from common import REVISIONS

from ways import load_backend

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache/controlled"
OUT = HERE / "results/checks"
SR, GAP_S = 16000, 1.0
MODELS = {"whisper-tiny": "openai/whisper-tiny", "whisper-base": "openai/whisper-base"}


def onset_s(wav):
    n = len(wav) // 320
    db = 10 * np.log10((wav[: n * 320].reshape(n, 320) ** 2).mean(1) + 1e-12)
    return float(np.argmax(db > db.max() - 30) * 0.02)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = list(pd.read_csv(HERE / "results/controlled/manifest.csv").id)
    pairs = [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]
    S = dict(gap_s=GAP_S, n_pairs=len(pairs), models={})
    for name, mid in MODELS.items():
        be = load_backend("hf-whisper", mid, revision=REVISIONS[mid])
        rows = []
        for ia, ib in pairs:
            a, _ = sf.read(CACHE / f"{ia}_clean.wav", dtype="float32")
            b, _ = sf.read(CACHE / f"{ib}_clean.wav", dtype="float32")
            joined = np.concatenate([a, np.zeros(int(GAP_S * SR), np.float32), b])
            ta, tj = be.transcribe(a), be.transcribe(joined)
            ka = ta.n_tokens
            if tj.n_tokens <= ka or not np.array_equal(tj.token_ids()[:ka], ta.token_ids()):
                continue
            end_a, start_b = len(a) / SR, len(a) / SR + GAP_S
            t = np.asarray(tj.times, float)
            kw = max(k for k in range(ka) if any(c.isalnum() for c in tj.pieces[k]))
            rows.append(dict(a=ia, b=ib, last_a=t[kw] - end_a, stop_a=t[ka - 1] - end_a if ka - 1 > kw else np.nan, first_b=t[ka] - start_b, first_b_vs_onset=t[ka] - start_b - onset_s(b),
                             first_a_vs_onset=t[0] - onset_s(a), last_a_in_b=bool(t[kw] >= end_a),
                             first_b_in_b=bool(t[ka] >= start_b - 0.1)))
        d = pd.DataFrame(rows)
        S["models"][name] = dict(pairs_used=int(len(d)), last_a_word_after_end_of_a=int(d.last_a_in_b.sum()), first_b_token_inside_b=int(d.first_b_in_b.sum()),
                                 last_a_word_minus_end_of_a_s=[float(v) for v in np.percentile(d.last_a, [25, 50, 75])],
                                 final_stop_of_a_minus_end_of_a_s=[float(v) for v in np.nanpercentile(d.stop_a, [25, 50, 75])],
                                 first_b_minus_onset_of_b_s=[float(v) for v in np.percentile(d.first_b_vs_onset, [25, 50, 75])],
                                 first_a_minus_onset_of_a_s=[float(v) for v in np.percentile(d.first_a_vs_onset, [25, 50, 75])])
        print(name, json.dumps(S["models"][name]), flush=True)
    (OUT / "timestamps.json").write_text(json.dumps(S, indent=1))
    md = ["| model | pairs used | last word of A timed after A ends | first token of B timed inside B | last word of A minus end of A (s), quartiles | final stop of A minus end of A (s) | first B token minus B's onset (s) | first A token minus A's onset (s) |", "|---|---|---|---|---|---|---|---|"]
    for name, m in S["models"].items():
        q = lambda v: " / ".join(f"{x:+.2f}" for x in v)
        md.append(f"| {name} | {m['pairs_used']} | {m['last_a_word_after_end_of_a']} | {m['first_b_token_inside_b']} | {q(m['last_a_word_minus_end_of_a_s'])} | {q(m['final_stop_of_a_minus_end_of_a_s'])} | "
                  f"{q(m['first_b_minus_onset_of_b_s'])} | {q(m['first_a_minus_onset_of_a_s'])} |")
    (OUT / "timestamps.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
