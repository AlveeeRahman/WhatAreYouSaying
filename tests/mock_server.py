"""A minimal OpenAI-compatible speech server for testing the black-box backend without vLLM or
llama.cpp: POST /v1/audio/transcriptions (multipart file) and POST /v1/chat/completions with an
input_audio part, both answered by a local Hugging Face Whisper. Run:  python tests/mock_server.py 8765
"""
import base64
import io
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf


class Handler(BaseHTTPRequestHandler):
    backend = None

    def log_message(self, *a):
        pass

    def _wav(self, data: bytes) -> np.ndarray:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == 16000
        return wav.mean(1) if wav.ndim > 1 else wav

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path.endswith("/audio/transcriptions"):
            m = re.search(rb"Content-Type: audio/wav\r\n\r\n(.*?)\r\n--", body, re.S)
            text = self.backend.transcribe(self._wav(m.group(1))).text
            out = {"text": text}
        elif self.path.endswith("/chat/completions"):
            req = json.loads(body)
            part = next(p for p in req["messages"][-1]["content"] if p["type"] == "input_audio")
            text = self.backend.transcribe(self._wav(base64.b64decode(part["input_audio"]["data"]))).text
            out = {"id": "mock", "object": "chat.completion", "created": 0, "model": req.get("model", ""),
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port: int, model_id: str = "openai/whisper-tiny"):
    from ways.backends.hf_whisper import WhisperBackend

    Handler.backend = WhisperBackend(model_id)
    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"mock server on {port} with {model_id}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8765, sys.argv[2] if len(sys.argv) > 2 else "openai/whisper-tiny")
