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


def sampling_parameters(model, *, thinking=True, temperature=None):
    """Resolved request settings, also recorded in generation manifests."""
    from vllm_reasoning_utils import resolve_sampling_defaults, resolve_chat_template_kwargs
    from matchminer_ai.llm.sampling import vendor_defaults

    settings = resolve_sampling_defaults(model, fallback={"temperature": 0.2},
                                        enable_thinking=thinking, temperature=temperature)
    extra_keys = {"top_k", "min_p", "repetition_penalty"}
    params = {key: value for key, value in settings.items() if key not in extra_keys}
    extra = {key: value for key, value in settings.items() if key in extra_keys}
    _, template = vendor_defaults(model, template=resolve_chat_template_kwargs(model, enable_thinking=thinking))
    extra["chat_template_kwargs"] = template
    return {**params, "extra_body": extra}


class TeacherPool:
    def __init__(self, urls=(), model=DEFAULT_TEACHER, *, registry=None, per_server=96,
                 max_tokens=16384, thinking=True, temperature=None, timeout=7200,
                 recovery_timeout=86400, trace_dir=None):
        self.pool = EndpointPool(urls, registry=registry, per_server=per_server,
            timeout=timeout, recovery_timeout=recovery_timeout,
            trace_dir=trace_dir,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.parameters = dict(model=model, max_tokens=max_tokens,
                               **sampling_parameters(model, thinking=thinking, temperature=temperature))

    def __call__(self, messages):
        return self.pool.request(messages=messages, **self.parameters)
