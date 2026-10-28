"""Release complete summary products while their large trace archive is publishing."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from retrieval_workspace import copy_verified, atomic_json, require_local_disk


def option(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def paths_for(command):
    data = Path(option(command, '--data-dir')).resolve()
    upstream = Path(option(command, '--pipeline-dir', str(Path(option(command, '--run-dir')) / 'upstream'))).resolve()
    public = upstream / 'summaries'
    scratch = Path(option(command, '--summary-scratch-dir', str(Path(tempfile.gettempdir()) / 'mmai-summaries'))).resolve()
    local = scratch / hashlib.sha256(str(public).encode()).hexdigest()[:24]
    return dict(public=public, local=local, metadata=data/'synthetic_patient_metadata.parquet',
                raw=data/'patient_summaries.parquet',
                attached=Path(option(command, '--patient-summaries', str(data/'patient_summaries_with_spaces.parquet'))))


def local_products_ready(paths, *, started_at):
    status = paths['public'] / 'storage_status.json'
    if not status.exists():
        return False
    state = json.loads(status.read_text())
    if state.get('at', 0) < started_at:
        return False
    if state.get('phase') == 'failed':
        raise RuntimeError('Summary publication failed: ' + str(state.get('error')))
    # These phases are written only after the summary callback has closed both
    # final Parquets. A completed checkpoint count alone is insufficient.
    if state.get('phase') not in ('archiving_checkpoints', 'publishing_archive'):
        return False
    if state.get('local_workspace') != str(paths['local']):
        raise ValueError('Summary readiness points to a different local workspace')
    return True


def publish_product(source, destination):
    # The original publisher can reach this same file concurrently. Separate
    # temporary names prevent interference; both publish identical closed bytes.
    staging = destination.with_name(destination.name + '.overlap-' + uuid.uuid4().hex)
    record = copy_verified(source, staging)
    staging.replace(destination)
    return record


def release_products(paths, receipt):
    from prepare_synthetic_training import prepare_summaries
    local = paths['local']; require_local_disk(local.parent)
    snapshot = local.parent / (local.name + '-retrieval-inputs')
    snapshot.mkdir(parents=True, exist_ok=True)
    source = local/'outputs/patient_summaries.parquet'
    raw = snapshot/'patient_summaries.parquet'
    if raw.exists():
        if not source.exists():
            return None
        # The snapshot is a hard link or copy of the source; compare sizes.
        if raw.stat().st_size != source.stat().st_size:
            raise ValueError('Retained retrieval summary snapshot changed')
    if not raw.exists():
        try:
            os.link(source, raw)
        except FileNotFoundError:
            # Cleanup means ordinary publication won the race; the caller must
            # wait for and verify the normal pipeline's completion instead.
            return None
        except OSError:
            copy_verified(source, raw)
    metadata = snapshot/'synthetic_patient_metadata.parquet'
    metadata_record = copy_verified(paths['metadata'], metadata)
    attached = snapshot/'patient_summaries_with_spaces.parquet'
    result = prepare_summaries(raw, metadata, attached)
    raw_record = publish_product(raw, paths['raw'])
    attached_record = publish_product(attached, paths['attached'])
    value = dict(at=time.time(), local_snapshot=str(snapshot), patients=result['patients'],
                 raw=dict(path=str(paths['raw']), **raw_record),
                 attached=dict(path=str(paths['attached']), **attached_record),
                 metadata=dict(path=str(paths['metadata']), **metadata_record),
                 archive_publication_pending=True)
    atomic_json(receipt, value)
    return value


def verify_published_products(receipt):
    if not Path(receipt).exists():
        return
    value = json.loads(Path(receipt).read_text())
    # The normal pipeline may republish these files, so compare byte sizes.
    for name in ('raw', 'attached', 'metadata'):
        record = value[name]
        if Path(record['path']).stat().st_size != record['bytes']:
            raise ValueError('Published summary product differs from retrieval input: ' + name)
    value.update(archive_publication_pending=False, publication_verified_at=time.time())
    atomic_json(receipt, value)
