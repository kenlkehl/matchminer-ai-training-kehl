"""Separate, resumable EmbeddingGemma retrieval/labeling after its first fit."""
import copy
import fcntl
import json
from pathlib import Path

from scripts.gcp_note_fleet import save


def argument(command, flag):
    return command[command.index(flag) + 1]


def replace_argument(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def configure(config, root, allowed_urls):
    """Preserve round one and freeze independent commands for the next pass."""
    result = copy.deepcopy(config)
    root = Path(root) / 'round2_embeddinggemma2'
    stages = result['stages']
    model = argument(stages['training']['command'], '--output-model')
    corpus = argument(stages['mining_dense']['command'], '--corpus')
    candidates, labels, pairs = root/'candidates', root/'labels', root/'pair_labels'
    registry = root/'servers.json'
    result['dense_round_two'] = dict(
        model=model, corpus=corpus, candidates=str(candidates), labels=str(labels),
        pairs=str(pairs), registry=str(registry), allowed_urls=list(allowed_urls),
        cloud_enabled=False, after_labels='review', colbert_labeling_deferred=True)
    result['train_dense_before_colbert_labels'] = True
    command = list(stages['mining_dense']['command'])
    replace_argument(command, '--model', model)
    replace_argument(command, '--output', candidates)
    if '--revision' in command:
        index = command.index('--revision')
        del command[index:index+2]
    stages['mining_dense_round2'] = dict(command=command, outputs=[str(candidates/'complete.json')])
    command = list(stages['labeling_dense']['command'])
    for flag, value in [('--candidates', candidates/'candidates.jsonl'), ('--output', labels),
                        ('--server_urls_file', registry),
                        ('--scheduler-metrics-file', root/'concurrency_status.json')]:
        replace_argument(command, flag, value)
    stages['labeling_dense_round2'] = dict(command=command, outputs=[
        str(labels/name) for name in ['scores_ready.json', 'scores.jsonl', 'status.json']])
    command = list(stages['project_dense_labels']['command'])
    for flag, value in [('--embeddinggemma', candidates), ('--labels', labels), ('--output', pairs)]:
        replace_argument(command, flag, value)
    stages['project_dense_round2'] = dict(command=command, outputs=[str(pairs/'complete.json'), *[
        str(pairs/'embeddinggemma2'/name) for name in ['complete.json', 'pairs.parquet',
        'patient_to_cohort.parquet', 'cohort_to_patient.parquet', 'scores.jsonl']]])
    return result


def validate_stage(config, name):
    plan = config['dense_round_two']
    if plan['cloud_enabled'] or not plan['allowed_urls']:
        raise ValueError('Round two requires an explicit local/LAN-only pool')
    command = config['stages'][name]['command']
    if name == 'mining_dense_round2':
        model = argument(config['stages']['training']['command'], '--output-model')
        if argument(command, '--model') != model or '--revision' in command:
            raise ValueError('Round-two retrieval must use the round-one trained model')
        if argument(command, '--output') != plan['candidates']:
            raise ValueError('Round-two candidates must have a separate destination')
    elif name == 'labeling_dense_round2':
        if (argument(command, '--server_urls_file') != plan['registry']
                or argument(command, '--candidates') != str(Path(plan['candidates'])/'candidates.jsonl')
                or argument(command, '--output') != plan['labels']):
            raise ValueError('Round-two labeling must use its own candidates, labels and local/LAN registry')


def sync_registry(config):
    """Follow enabled authorized LAN teachers without admitting cloud tunnels."""
    plan = config['dense_round_two']
    if plan['cloud_enabled']:
        raise ValueError('Cloud labeling is disabled for round two')
    registry, destination = Path(config['registry']), Path(plan['registry'])
    if registry.resolve() == destination.resolve():
        raise ValueError('Round two needs a separate local/LAN registry')
    allowed = set(plan['allowed_urls'])
    with registry.with_name('servers.fleet.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        source = json.loads(registry.read_text())
    servers = [row for row in source['servers']
               if row['url'] in allowed and not row.get('fleet_id')]
    value = dict(servers=servers, ready=bool(servers), cloud_enabled=False)
    if not destination.exists() or json.loads(destination.read_text()) != value:
        save(destination, value)
    return value
