"""Live teacher pool shared by distillation and criterion preparation."""
import json
import os
from pathlib import Path

from matchminer_ai.llm.endpoint_pool import EndpointPool, MAX_POOL_WORKERS

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


def worker_count(concurrency):
    if concurrency < 0:
        raise ValueError('Concurrency must be zero (automatic) or positive')
    return concurrency or MAX_POOL_WORKERS


class TeacherPool:
    def __init__(self, urls=(), model=DEFAULT_TEACHER, *, registry=None, per_server=96,
                 max_tokens=16384, thinking=True, temperature=0.2, timeout=7200,
                 recovery_timeout=86400):
        self.pool = EndpointPool(urls, registry=registry, per_server=per_server,
            timeout=timeout, recovery_timeout=recovery_timeout,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.parameters = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                               extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})

    def __call__(self, messages):
        return self.pool.request(messages=messages, **self.parameters)
