"""WhatAreYouSaying (WAYS): which audio did a speech-to-text model use for each token it wrote,
and was that audio clean or degraded?"""
from .attribution import Attribution, explain
from .backends import load_backend
from .erasure import PatchGrid, erase

__version__ = "0.2.1"
__all__ = ["Attribution", "explain", "load_backend", "PatchGrid", "erase", "__version__"]
