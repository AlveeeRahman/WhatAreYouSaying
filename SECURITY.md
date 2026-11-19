# Security

WAYS runs locally on audio you give it. This page says what it does with the network and your
data, and how to report a problem.

## Reporting a vulnerability

Please report privately through GitHub: open the repository's Security tab and choose "Report a
vulnerability". Do not open a public issue for anything that could be exploited before it is fixed.

## What the code does and does not do

- Makes no network request of its own except the ones you ask for: the Hugging Face Hub on the
  first load of a model (pinned snapshot commits in `experiments/common.py`), AADD's checkpoint
  download from `huggingface.co/gertelrina/aadd` on first use, Edinburgh DataShare when you run
  `experiments/demand.py`, and the OpenAI-compatible server you point the `openai-compat` backend at.
- Sends audio to that server, and nowhere else, when you choose the `openai-compat` backend. The
  white-box backends keep the audio in the process.
- Reads only the files you name and the data folder you point it at (`WAYS_DATA`). It never writes
  outside its output path, the experiment folders and the Hugging Face and AADD caches.
- Stores no credentials. The `openai` client reads `OPENAI_API_KEY` from the environment if your
  server needs one; WAYS never prints or logs it.
- Writes no absolute path, user name or machine layout into the files it produces.

## Known limits

- Model weights and the AADD checkpoints are downloaded over HTTPS from the Hub and verified by
  the Hub's own mechanisms; AADD's checkpoints are not pinned to a revision (their sha256 values
  are recorded in `docs/reproducibility.md`).
- `experiments/demand.py` verifies the SHA-256 of the archives it downloads; the LibriSpeech
  archives in `docs/reproducibility.md` are not checksummed by the scripts.
- A malicious audio file is parsed by `soundfile` (libsndfile) and `librosa`; WAYS adds no checks
  beyond theirs.
