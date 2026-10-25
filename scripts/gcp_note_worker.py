#!/usr/bin/env python3
"""VM-local per-GPU Gemma workers and an expiring controller lease."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/opt/mmai-gemma-note-fleet')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def status(stage, **kwargs):
    value = dict(stage=stage, at=datetime.now(timezone.utc).isoformat(), **kwargs)
    save(ROOT / 'status.json', value)
    print(json.dumps(value), flush=True)


def request(gpu, route, body=None):
    req = urllib.request.Request(f'http://127.0.0.1:{8000 + gpu}{route}',
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180 if body else 5) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def ready(gpu, config):
    try:
        request(gpu, '/health')
        return any(m['id'] == config['served_model'] and m.get('root') == config['model']
                   and m.get('max_model_len', 0) >= config['context']
                   for m in request(gpu, '/v1/models')['data'])
    except Exception:
        return False


def smoke(gpu, config):
    body = dict(model=config['served_model'], messages=[dict(role='user', content='What is 7 + 5? Return only the number.')],
        temperature=1.0, top_p=.95, top_k=64, max_tokens=2048,
        chat_template_kwargs={'enable_thinking': True})
    response = request(gpu, '/v1/chat/completions', body)
    choice = response['choices'][0]
    message = choice['message']
    valid = (choice['finish_reason'] == 'stop' and '12' in (message.get('content') or '')
             and bool(message.get('reasoning') or message.get('reasoning_content')))
    save(ROOT / f'smoke-{gpu}.json', dict(passed=valid, request=body, response=response,
                                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    if not valid:
        raise ValueError(f'GPU {gpu} failed reasoning/completion smoke')


def launch(gpu, config, hub):
    name = f'mmai-gemma-note-gpu{gpu}'
    exists = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if exists.returncode == 0:
        old = json.loads(exists.stdout)[0]
        if old['Config']['Labels'].get('mmai.config') == config['config_hash']:
            run(['docker', 'start', name])
            return
        run(['docker', 'rm', '-f', name])
    cache = ROOT / f'cache/gpu-{gpu}'
    cache.mkdir(parents=True, exist_ok=True)
    command = ['docker', 'run', '-d', '--name', name, '--gpus', f'device={gpu}',
        '--label', 'mmai.config=' + config['config_hash'], '--ipc=host', '--network=host',
        '--restart=unless-stopped', '--ulimit', 'memlock=-1', '--ulimit', 'nofile=1048576:1048576',
        '--log-opt', 'max-size=100m', '--log-opt', 'max-file=3',
        '-v', str(hub) + ':/root/.cache/huggingface/hub:ro',
        '-v', str(cache) + ':/root/.cache/vllm', '-e', 'HF_HUB_OFFLINE=1',
        '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
        '-e', 'OMP_NUM_THREADS=8', '-e', 'MAX_JOBS=4', '-e', 'VLLM_ENGINE_READY_TIMEOUT_S=3600',
        config['image'], config['model'], '--revision', config['revision'],
        '--host', '127.0.0.1', '--port', str(8000 + gpu), '--served-model-name', config['served_model'],
        '--tensor-parallel-size', '1', '--quantization', 'modelopt', '--language-model-only',
        '--max-model-len', str(config['context']), '--max-num-seqs', str(config['max_num_seqs']),
        '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '0.90',
        '--enable-prefix-caching', '--reasoning-parser', 'gemma4', '--no-enable-log-requests']
    save(ROOT / f'command-{gpu}.json', command)
    run(command)


def bootstrap():
    config = json.loads((ROOT / 'config.json').read_text())
    status('preparing_runtime')
    count = len(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines())
    if count != 8:
        raise ValueError(f'Expected eight GPUs, found {count}')
    if subprocess.run(['docker', 'image', 'inspect', config['image']], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        run(['docker', 'pull', config['image']], timeout=1800)
    hub = ROOT / 'hf-hub'
    hub.mkdir(exist_ok=True)
    for candidate in (Path('/home/kenneth_kehl/.cache/huggingface/hub'), Path('/root/.cache/huggingface/hub')):
        snapshot = candidate / ('models--' + config['model'].replace('/', '--')) / 'snapshots' / config['revision']
        if (snapshot / 'model.safetensors.index.json').exists():
            index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
            if all((snapshot / f).exists() for f in set(index['weight_map'].values())):
                hub = candidate
                break
    status('downloading_and_verifying_model', hub=str(hub))
    code = ('from huggingface_hub import snapshot_download; from transformers import AutoTokenizer; import hashlib,json; '
            f'p=snapshot_download({config["model"]!r}, revision={config["revision"]!r}, max_workers=8); '
            't=AutoTokenizer.from_pretrained(p); '
            'h=hashlib.sha256(str(t.chat_template).encode()).hexdigest(); '
            f'assert h == {config["chat_template_sha256"]!r}, "Tokenizer template differs from the active generation run"; '
            'print(json.dumps(dict(snapshot=p,chat_template_sha256=h)))')
    run(['docker', 'run', '--rm', '--name', 'mmai-gemma-note-download',
         '-v', str(hub) + ':/root/.cache/huggingface/hub', '--entrypoint', 'python3', config['image'], '-c', code], timeout=1800)
    status('starting_canary')
    launch(0, config, hub)
    deadline = time.monotonic() + 1800
    while not ready(0, config):
        if time.monotonic() > deadline:
            raise TimeoutError('Gemma canary startup timed out')
        time.sleep(10)
    smoke(0, config)
    status('starting_remaining_workers', healthy_gpus=[0])
    for gpu in range(1, 8):
        launch(gpu, config, hub)
    passed = {0}
    while len(passed) < 8:
        for gpu in range(1, 8):
            if gpu not in passed and ready(gpu, config):
                smoke(gpu, config)
                passed.add(gpu)
        status('warming_workers', healthy_gpus=sorted(passed))
        if time.monotonic() > deadline:
            raise TimeoutError('Not all GPU workers became healthy')
        time.sleep(10)
    status('ready', healthy_gpus=sorted(passed), config_hash=config['config_hash'])


def guard():
    """Stop the VM if the active local controller can no longer renew its lease."""
    while True:
        try:
            lease = json.loads((ROOT / 'lease.json').read_text())
            valid = float(lease['expires_at']) > time.time()
        except Exception:
            valid = False
        if not valid:
            save(ROOT / 'shutdown.json', dict(reason='controller lease expired', at=time.time()))
            run(['systemctl', 'poweroff'])
            return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard', action='store_true')
    if parser.parse_args().guard:
        guard()
    else:
        try:
            bootstrap()
        except Exception as error:
            status('failed', error=repr(error))
            raise
