"""Prepare SSD training pairs from complete scores while trace publication runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from colbert_experiments.local_teacher import process_identity
from colbert_experiments.score_publication import ready_scores, input_fingerprints
from retrieval_workspace import atomic_json, copy_verified, require_local_disk, require_free_space
from training_workspace import publish_tree, tree_manifest, verify_tree
from trial_cohort_format import file_stamp


def argument(command, flag):
    return command[command.index(flag)+1]


def configure(config, root, *, scratch='/tmp/mmai-dense-training'):
    """Keep public destinations while moving pair construction and fitting to SSD."""
    config = copy.deepcopy(config)
    root = Path(root)
    original = config['stages']['project_dense_labels']
    labels = argument(original['command'], '--labels')
    public_pairs = argument(original['command'], '--output')
    work = Path(scratch)/hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]
    label_work = Path('/tmp/mmai-labels')/hashlib.sha256(str(Path(labels).resolve()).encode()).hexdigest()[:24]
    storage_status = Path(labels)/'storage_status.json'
    if storage_status.exists():
        label_work = Path(json.loads(storage_status.read_text())['workspace'])
    config['local_dense_training'] = dict(workspace=str(work), labels=labels,
        corpus=argument(original['command'], '--corpus'),
        candidates=argument(original['command'], '--embeddinggemma'),
        label_workspace=str(label_work), label_process_file=str(root/'processes/labeling_dense.json'),
        public_pairs=public_pairs)
    command = [original['command'][0], str(Path(config['repo'])/'dense_training_preparation.py')]
    config['stages']['project_dense_labels'] = dict(
        command=command+['prepare', '--config', str(root/'config.json')],
        outputs=[str(work/'snapshot/complete.json'), str(work/'pairs/complete.json'),
                 str(work/'pairs/embeddinggemma2/pairs.parquet')])
    config['stages']['publish_dense_training_pairs'] = dict(
        command=command+['publish', '--config', str(root/'config.json')],
        outputs=[*original['outputs'], str(Path(public_pairs)/'storage_complete.json')])
    config['training_pair_labels_root'] = str(work/'pairs')
    training = config['stages']['training']['command']
    training[training.index('--input-parquet')+1] = str(work/'pairs/embeddinggemma2/pairs.parquet')
    if '--no-local-training-storage' in training:
        training.remove('--no-local-training-storage')
    if '--local-training-storage' not in training:
        training.append('--local-training-storage')
    checkpoints = Path(argument(training, '--ckpt-dir'))
    config['stages']['training']['outputs'].append(str(checkpoints/'training_storage_complete.json'))
    return config


def score_source(plan):
    """Only the registered labeler may release unpublished local training scores."""
    public = Path(plan['labels'])
    if (public/'scores_ready.json').exists():
        return dict(labels=public, corpus=Path(plan['corpus']), candidates=Path(plan['candidates'])/'candidates.jsonl')
    work = Path(plan['label_workspace'])
    status = work/'state/status.json'
    if not status.exists() or not (work/'state/scores_ready.json').exists():
        return None
    expected = json.loads(Path(plan['label_process_file']).read_text())
    current = process_identity(expected['pid'])
    if current != expected:
        raise ValueError('Local training-score publisher identity changed')
    for flag, value in [('--corpus', plan['corpus']), ('--candidates', str(Path(plan['candidates'])/'candidates.jsonl')),
                        ('--output', plan['labels'])]:
        if argument(expected['command'], flag) != value:
            raise ValueError('Registered labeler does not own these training inputs')
    identity = json.loads((work/'identity.json').read_text())
    if identity['public'] != str(public.resolve()):
        raise ValueError('Local training-score destination changed')
    state = json.loads(status.read_text())
    if (state.get('pid') != expected['pid'] or state.get('inference_complete') is not True
            or state.get('failures') or state.get('scored_pairs') != state.get('total_pairs')
            or not state.get('total_pairs')):
        return None
    return dict(labels=work/'state', corpus=work/'inputs/corpus', candidates=work/'inputs/candidates.jsonl')


def prepare(plan):
    root = Path(plan['workspace'])
    root.mkdir(parents=True, exist_ok=True)
    require_local_disk(root)
    require_free_space(root, reserve_gb=32)
    snapshot = root/'snapshot'
    receipt_path = snapshot/'complete.json'
    count = json.loads((Path(plan['candidates'])/'complete.json').read_text())['pairs']
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved['plan'] != plan:
            raise ValueError('SSD training preparation settings changed')
        verify_tree(snapshot, saved['products'])
    else:
        source = score_source(plan)
        if source is None:
            raise ValueError('Complete training scores are not available yet')
        # The complete-score receipt is the training gate. Large reasoning
        # archives remain independently supervised and are joined before reuse.
        ready = ready_scores(source['labels'], corpus=source['corpus'],
                             candidates=source['candidates'], expected_pairs=count)
        if ready is None:
            raise ValueError('Missing complete score publication')
        public_candidates = Path(plan['candidates'])/'candidates.jsonl'
        public_inputs = input_fingerprints(plan['corpus'], public_candidates)
        # The labeler may have fingerprinted staged copies; copies are compared by size.
        for name, expected in ready['inputs'].items():
            if public_inputs[name]['bytes'] != expected['bytes']:
                raise ValueError('Local scores differ from the intended published corpus/candidates')
        candidates_stamp = json.loads((Path(plan['candidates'])/'complete.json').read_text())['outputs']['candidates.jsonl']
        if file_stamp(public_candidates) != candidates_stamp:
            raise ValueError('Published candidate completion identity changed')
        paths = [(p, snapshot/'corpus'/p.name) for p in source['corpus'].iterdir() if p.is_file()]
        paths += [(source['candidates'], snapshot/'candidates/candidates.jsonl'),
                  (Path(plan['candidates'])/'complete.json', snapshot/'candidates/complete.json')]
        paths += [(source['labels']/name, snapshot/'labels'/name)
                  for name in ['scores_ready.json', 'scores.jsonl', 'labeling_manifest.json']]
        for src, dst in paths:
            require_free_space(root, src.stat().st_size, 32)
            copy_verified(src, dst)
        ready_scores(snapshot/'labels', corpus=snapshot/'corpus',
                     candidates=snapshot/'candidates/candidates.jsonl', expected_pairs=count)
        atomic_json(receipt_path, dict(plan=plan, products=tree_manifest(snapshot), pairs=count,
                                      training_scores_complete=True, trace_publication_independent=True))
    from trialspace_comparison import project_labels
    project_labels(snapshot/'corpus', snapshot/'candidates', snapshot/'labels',
                   {'embeddinggemma2': snapshot/'candidates'}, root/'pairs',
                   reasoning_source=str(Path(plan['labels'])/'responses'),
                   source_origins={'embeddinggemma2': plan['candidates']})


def publish(plan):
    source = Path(plan['workspace'])/'pairs'
    if not (source/'complete.json').exists():
        raise ValueError('Training pair projection has not completed')
    products = publish_tree(source, Path(plan['public_pairs']))
    atomic_json(Path(plan['public_pairs'])/'storage_complete.json', dict(
        complete=True, workspace=str(source), products=products))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'publish'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())['local_dense_training']
    (prepare if args.action == 'prepare' else publish)(plan)


if __name__ == '__main__':
    main()
