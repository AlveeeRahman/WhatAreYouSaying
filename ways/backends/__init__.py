from .base import Backend, Transcript


def load_backend(kind: str, model_id: str | None = None, **kw) -> Backend:
    """kind: hf-whisper | hf-moonshine | openai-compat"""
    if kind == "hf-whisper":
        from .hf_whisper import WhisperBackend
        return WhisperBackend(model_id or "openai/whisper-base", **kw)
    if kind == "hf-moonshine":
        from .hf_moonshine import MoonshineBackend
        return MoonshineBackend(model_id or "UsefulSensors/moonshine-base", **kw)
    if kind == "openai-compat":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model_id=model_id or "openai/whisper-base", **kw)
    raise ValueError(f"unknown backend {kind!r}; choose hf-whisper, hf-moonshine or openai-compat")


__all__ = ["Backend", "Transcript", "load_backend"]
