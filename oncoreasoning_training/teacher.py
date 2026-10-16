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
                 max_tokens=16384, thinking=True, temperature=None, timeout=7200,
                 recovery_timeout=86400):
        self.pool = EndpointPool(urls, registry=registry, per_server=per_server,
            timeout=timeout, recovery_timeout=recovery_timeout,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        from matchminer_ai.llm.model_profiles import resolve_model_profile
        profile = resolve_model_profile(model)
        # Preserve historical settings for existing teachers. Nemotron's new
        # text profile is opt-in through its model ID; explicit temperature wins.
        params, extra = {}, {}
        if profile and profile.name == "nemotron-3.5-super-vl":
            params, extra = dict(profile.request_params), dict(profile.extra_body)
        params["temperature"] = temperature if temperature is not None else params.get("temperature", 0.2)
        extra["chat_template_kwargs"] = {"enable_thinking": thinking}
        self.parameters = dict(model=model, max_tokens=max_tokens, **params, extra_body=extra)

    def __call__(self, messages):
        return self.pool.request(messages=messages, **self.parameters)
