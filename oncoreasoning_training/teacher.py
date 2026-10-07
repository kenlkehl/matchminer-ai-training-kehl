"""Small OpenAI-compatible pool shared by distillation and criterion preparation."""
import json
import os
from pathlib import Path
import threading

DEFAULT_TEACHER = "nvidia/Gemma-4-31B-IT-NVFP4"


def server_urls(url=None, file=None):
    if file:
        data = json.loads(Path(file).read_text())
        entries = data.get("servers", data.get("server_urls", [])) if isinstance(data, dict) else data
        urls = [entry["url"] if isinstance(entry, dict) else entry for entry in entries]
    else:
        urls = [url] if url else []
    if not urls or any(not value.startswith(("http://", "https://")) for value in urls):
        raise ValueError("Supply an explicit teacher endpoint or ready server pool")
    return list(dict.fromkeys(value.rstrip('/') for value in urls))


class TeacherPool:
    def __init__(self, urls, model=DEFAULT_TEACHER, *, max_tokens=16384, thinking=True, temperature=0.2, timeout=600):
        from openai import OpenAI
        self.clients = [OpenAI(base_url=url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout, max_retries=0) for url in urls]
        self.parameters = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                               extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
        self.lock, self.index = threading.Lock(), 0

    def __call__(self, messages):
        with self.lock:
            client = self.clients[self.index % len(self.clients)]
            self.index += 1
        return client.chat.completions.create(messages=messages, **self.parameters)
