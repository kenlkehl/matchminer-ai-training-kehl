"""Reuse unchanged encodings when source partitions are merged, never rankings."""
import json
import math
from pathlib import Path

from colbert_experiments.data import digest, load_corpus, write_json
from retrieval_workspace import copy_verified
from synthetic_pool import pooled_records, require_full_pool
from trial_cohort_format import file_stamp


def stage_full_pool_encodings(source, destination, old_corpus, new_corpus, *, kind):
    source, destination = Path(source), Path(destination)
    old = load_corpus(old_corpus)
    new = load_corpus(new_corpus)
    require_full_pool(*new)
    manifest = json.loads((source/('manifest.json' if kind == 'dense' else 'mining_manifest.json')).read_text())
    for name, old_rows, new_rows in zip(('patients','cohorts'),old,new):
        if digest(old_rows) != manifest[name] or pooled_records(old_rows) != new_rows:
            raise ValueError('Encoding reuse requires identical entity order, text and provenance: '+name)
    fields = (['model','revision','prompts','max_tokens','dimension'] if kind == 'dense' else
              ['model','encoding_device_kind','token_storage_dtype','document_batch_size'])
    expected = {name:manifest[name] for name in fields}
    expected.update(patients=digest(new[0]),cohorts=digest(new[1]))
    if kind == 'dense':
        complete = json.loads((source/'complete.json').read_text())
        files = [Path('patients.npy'),Path('cohorts.npy')]
        stamps = {str(p):complete['outputs'][str(p)] for p in files}
    elif kind == 'colbert':
        if set(manifest['splits']) != {'train','val','test'}:
            raise ValueError('Source token cache must cover every original partition')
        size = manifest['document_batch_size']
        files = [Path('tokens')/f'{name}_{i:07d}.pt' for name,rows in zip(('patients','cohorts'),old)
                 for i in range(math.ceil(len(rows)/size))]
        stamps = {}
    else:
        raise ValueError('Unknown encoding cache kind')
    products = {str(p):copy_verified(source/p,destination/p,stamps.get(str(p))) for p in files}
    receipt = dict(kind=kind,source=str(source),original_corpus=str(old_corpus),expected=expected,
                   products=products,rankings_reused=False)
    write_json(destination/'encoding_import.json',receipt)
    return receipt


def validate_encoding_import(output, signature):
    path = Path(output)/'encoding_import.json'
    if not path.exists():
        return
    receipt = json.loads(path.read_text())
    if any(signature.get(name) != value for name,value in receipt['expected'].items()):
        raise ValueError('Imported encodings do not match the current model, text or encoding settings')
    for relative, record in receipt['products'].items():
        source = Path(output)/relative
        if not source.resolve().is_relative_to(Path(output).resolve()) or file_stamp(source) != record['stamp']:
            raise ValueError('Imported encoding cache changed: '+relative)
