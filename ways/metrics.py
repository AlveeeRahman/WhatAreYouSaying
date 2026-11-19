"""Summaries of an Attribution, and the statistics used to compare models.

All per-token summaries are computed from the token's non-negative mass
    m_k(P) = max(delta_k(P), 0) / sum_P max(delta_k(P), 0)
so they depend on the ranking of patches within a token and not on the token's confidence
(dividing by S_k(x), as in Li et al. eq. 1, is a per-token constant and cancels here).
"""
from __future__ import annotations

import re

import numpy as np

from .attribution import Attribution


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("—", " ").replace("–", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: str, hyp: str) -> dict:
    """Word error rate (S + D + I) / N against a reference, both normalised the same way."""
    import jiwer

    r, h = normalize_text(ref), normalize_text(hyp)
    n = len(r.split())
    if n == 0:
        return dict(S=0, D=0, I=len(h.split()), H=0, N=0, wer=np.nan)
    if not h:
        return dict(S=0, D=n, I=0, H=0, N=n, wer=1.0)
    o = jiwer.process_words(r, h)
    return dict(S=o.substitutions, D=o.deletions, I=o.insertions, H=o.hits, N=n,
                wer=(o.substitutions + o.deletions + o.insertions) / n)


def token_mass(att: Attribution) -> np.ndarray:
    """[K, B, T] non-negative mass per token, each token summing to 1 (nan rows if a token has no positive patch)."""
    m = np.clip(att.delta, 0, None).astype(np.float64)
    s = m.sum((1, 2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, m / s, np.nan)


def token_summary(att: Attribution, aadd_probs: np.ndarray | None = None, threshold: float = 0.5) -> list[dict]:
    """One row per scored unit (token or word)."""
    from .quality import flag_profile

    m = token_mass(att)
    K, B, T = m.shape
    f = flag_profile(aadd_probs, T, att.patch_s, threshold)
    fbar = float(f.mean()) if f is not None else np.nan
    rows = []
    for k in range(K):
        r = dict(k=k, piece=att.pieces[k], token_id=att.token_ids[k], surprisal0=float(att.surprisal0[k]),
                 t_tok=float(att.times[k]) if att.times is not None else np.nan,
                 flip_any=bool(att.flip[k].any()), flip_frac=float(att.flip[k].mean()),
                 max_delta=float(att.delta[k].max()), n_pos=int((att.delta[k] > 0).sum()))
        if np.isnan(m[k]).any():
            rows.append(r)
            continue
        prof = m[k].sum(0)            # [T]
        bands = m[k].sum(1)           # [B]
        top = int(prof.argmax())
        r.update(top_t=top, top_centre_s=(top + 0.5) * att.patch_s, top_share=float(prof.max()),
                 top_band=int(bands.argmax()), entropy_time=float(-(prof[prof > 0] * np.log(prof[prof > 0])).sum()))
        for b in range(B):
            r[f"band{b}"] = float(bands[b])
        if att.times is not None:
            r["dist_s"] = abs(r["top_centre_s"] - r["t_tok"])
        if f is not None and fbar > 0:
            r["mass_flagged"] = float((prof * f).sum())
            r["focus_ratio"] = r["mass_flagged"] / fbar
        rows.append(r)
    return rows


def clip_summary(att: Attribution, rows: list[dict], aadd_probs=None, threshold: float = 0.5) -> dict:
    from .quality import flagged_mask

    vals = lambda key: np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], float)
    out = dict(model=att.model, backend=att.backend, mode=att.mode, n_units=len(rows), duration=att.duration,
               n_patches=int(att.n_bands * att.n_t), seconds=att.seconds, n_forward=att.n_forward,
               mean_surprisal0=float(att.surprisal0.mean()) if len(att.surprisal0) else np.nan,
               flip_any_frac=float(np.mean([r["flip_any"] for r in rows])) if rows else np.nan,
               top_share_median=float(np.median(vals("top_share"))) if len(vals("top_share")) else np.nan)
    for b in range(att.n_bands):
        v = vals(f"band{b}")
        out[f"band{b}_mean"] = float(v.mean()) if len(v) else np.nan
    if len(vals("dist_s")):
        d = vals("dist_s")
        out.update(dist_median_s=float(np.median(d)), within_0p3=float((d <= 0.3).mean()), within_0p5=float((d <= 0.5).mean()))
    if aadd_probs is not None:
        n20 = int(np.ceil(att.duration / 0.02))
        fl = flagged_mask(aadd_probs[:n20], threshold)
        out["frac_flagged"] = float(fl.mean()) if len(fl) else np.nan
        fr = vals("focus_ratio")
        out["focus_ratio_mean"] = float(fr.mean()) if len(fr) else np.nan
        out["focus_ratio_median"] = float(np.median(fr)) if len(fr) else np.nan
        out["n_focus"] = int(len(fr))
    return out


# ------------------------------------------------------------------ statistics
def bootstrap_ci(x, B: int = 10000, seed: int = 8001, stat=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, ci=[np.nan, np.nan])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    boots = stat(x[idx], axis=1)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def paired_bootstrap(a, b, B: int = 10000, seed: int = 8001):
    """CI of mean(a - b) over paired clips (same clips, two models)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    r = bootstrap_ci(d, B, seed)
    r["n_pairs"] = int(ok.sum())
    return r
