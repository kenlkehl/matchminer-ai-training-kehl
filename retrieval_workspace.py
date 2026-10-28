"""Local retrieval workspaces with resumable, verified publication to shared storage."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from trial_cohort_format import file_stamp

INTERNAL = {'retrieval_status.json', 'retrieval_storage.json'}


def add_retrieval_storage_args(parser):
    parser.add_argument('--local-retrieval', action=argparse.BooleanOptionalAction, default=True,
                        help='Stage retrieval inputs, embeddings and checkpoints on local disk')
    parser.add_argument('--retrieval-scratch-dir', type=Path,
                        default=Path(tempfile.gettempdir()) / 'mmai-retrieval',
                        help='Local SSD workspace root; final --output stays on shared storage')
    parser.add_argument('--retrieval-min-free-gb', type=float, default=16,
                        help='Local disk reserve; insufficient space fails without discarding checkpoints')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require_free_space(path, required=0, reserve_gb=16):
    if reserve_gb < 0:
        raise ValueError('Retrieval disk reserve must be nonnegative')
    free = shutil.disk_usage(path).free
    if free < required + int(reserve_gb * 1024 ** 3):
        raise OSError(f'Insufficient local retrieval space at {path}: {free} bytes free; '
                      f'need {required} bytes plus {reserve_gb} GiB reserve')


def require_local_disk(path):
    """Do not silently put the hot working set back on a network mount."""
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/mounts').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace('\\040', ' '))
        if path.is_relative_to(mount):
            matches.append((len(mount.parts), fields[2]))
    filesystem = max(matches)[1] if matches else ''
    if filesystem in {'nfs', 'nfs4', 'cifs', 'smb3', '9p', 'ceph', 'fuse.sshfs', 'fuse.glusterfs'}:
        raise ValueError(f'Retrieval scratch must be local disk, not {filesystem}: {path}')
    return filesystem


def files(root):
    root = Path(root)
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Retrieval artifacts must not be symlinks: {path}')
        if path.is_file() and path.name not in INTERNAL and not path.name.endswith(('.tmp', '.lock')):
            result.append(path)
    return result


def copy_verified(source, target, expected=None):
    """Copy sequentially and fsync; check byte sizes before atomic rename.

    ``expected`` is the source's recorded file_stamp. The copy keeps the
    source's modification time, so it has the same stamp as its source and
    stamp-based completion records stay valid after publication.
    """
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    before = source.stat()
    if expected is not None and file_stamp(source) != expected:
        raise ValueError(f'Retrieval copy source changed: {source} -> {target}')
    with source.open('rb') as incoming, temporary.open('wb') as outgoing:
        while block := incoming.read(8 * 1024 ** 2):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    after = source.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or temporary.stat().st_size != before.st_size):
        raise ValueError(f'Retrieval copy verification failed: {source} -> {target}')
    os.utime(temporary, ns=(before.st_atime_ns, before.st_mtime_ns))
    temporary.replace(target)
    return dict(bytes=before.st_size, stamp=file_stamp(target))


class Workspace:
    def __init__(self, args):
        self.args = args
        self.destination, self.corpus = Path(args.output).resolve(), Path(args.corpus).resolve()
        self.root = Path(getattr(args, 'retrieval_scratch_dir', None) or
                         Path(tempfile.gettempdir()) / 'mmai-retrieval').resolve()
        self.reserve = getattr(args, 'retrieval_min_free_gb', 16)
        self.key = hashlib.sha256(str(self.destination).encode()).hexdigest()[:24]
        self.work = self.root / self.key
        if (self.root.is_relative_to(self.destination) or self.destination.is_relative_to(self.work)
                or self.corpus.is_relative_to(self.work)
                or self.corpus.is_relative_to(self.destination)
                or self.destination.is_relative_to(self.corpus)):
            raise ValueError('Retrieval scratch, source and destination must be separate')
        self.lock = self.destination_lock = None
        self.owns_lock = False

    def status(self, phase, **values):
        atomic_json(self.destination / 'retrieval_status.json', dict(
            phase=phase, at=time.time(), pid=os.getpid(), workspace=str(self.work), destination=str(self.destination), **values))
        print(f'[retrieval storage] {phase}: {self.work} -> {self.destination}', flush=True)

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        filesystem = require_local_disk(self.root)
        self.lock = (self.root / (self.key + '.lock')).open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.destination.mkdir(parents=True, exist_ok=True)
        self.destination_lock = (self.destination / 'retrieval.lock').open('a')
        fcntl.flock(self.destination_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owns_lock = True
        source_files = files(self.corpus)
        if not source_files:
            raise ValueError('Retrieval corpus is empty')
        inputs = {str(p.relative_to(self.corpus)): dict(bytes=p.stat().st_size, stamp=file_stamp(p))
                  for p in source_files}
        options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()
                   if k not in {'retrieval_scratch_dir', 'retrieval_min_free_gb', 'local_retrieval'}}
        identity = dict(version=1, corpus=str(self.corpus), destination=str(self.destination),
                        inputs=inputs, options=options)
        identity_path = self.work / 'workspace.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(f'Retrieval inputs/settings changed; retained scratch: {self.work}')
        require_free_space(self.root, sum(x['bytes'] for x in inputs.values()), self.reserve)
        self.work.mkdir(exist_ok=True)
        atomic_json(identity_path, identity)
        self.status('staging', filesystem=filesystem)
        for source in source_files:
            relative = source.relative_to(self.corpus)
            target = self.work / 'corpus' / relative
            expected = inputs[str(relative)]
            # workspace.json already pins the source stamps; a staged copy is checked by size.
            if target.exists() and target.stat().st_size == expected['bytes']:
                continue
            copy_verified(source, target, expected['stamp'])
        output = self.work / 'output'
        output.mkdir(exist_ok=True)
        # Recover previously published partial work when the local SSD was lost.
        # Existing local checkpoints are authoritative and are never overwritten.
        for source in files(self.destination):
            target = output / source.relative_to(self.destination)
            if not target.exists():
                require_free_space(self.root, source.stat().st_size, self.reserve)
                copy_verified(source, target)
        self.status('retrieving')
        return str(self.work / 'corpus'), str(output)

    def publish(self):
        output = self.work / 'output'
        marker = output / 'complete.json'
        if not marker.exists():
            raise ValueError('Retrieval did not produce a completion marker')
        completed = json.loads(marker.read_text())
        for name, expected in completed['outputs'].items():
            path = output / name
            if not path.resolve().is_relative_to(output.resolve()) or file_stamp(path) != expected:
                raise ValueError('Invalid completed retrieval artifact: ' + name)
        paths = files(output)
        self.status('publishing', files=len(paths), bytes=sum(p.stat().st_size for p in paths))
        # Completion is the publication gate: downstream labeling cannot consume
        # a partial result. All token shards/checkpoints are copied back as well.
        def one(source):
            relative = source.relative_to(output)
            return str(relative), copy_verified(source, self.destination / relative)
        with ThreadPoolExecutor(max_workers=4) as pool:
            products = dict(pool.map(one, [p for p in paths if p != marker]))
        marker_record = dict(bytes=marker.stat().st_size, stamp=file_stamp(marker))
        products['complete.json'] = marker_record
        atomic_json(self.destination / 'retrieval_storage.json', dict(
            version=1, original_corpus=str(self.corpus), destination=str(self.destination),
            workspace=str(self.work), verified_at=time.time(), products=products))
        copy_verified(marker, self.destination / 'complete.json', marker_record['stamp'])
        self.status('complete', files=len(products), bytes=sum(p['bytes'] for p in products.values()))
        shutil.rmtree(self.work)

    def close(self):
        if self.destination_lock is not None:
            self.destination_lock.close()
        if self.lock is not None:
            self.lock.close()


def run_retrieval(args, callback, *, distributed=None):
    """Run unchanged mining on local paths; rank zero stages/publishes for one host."""
    if not getattr(args, 'local_retrieval', True):
        return callback(args)
    rank, world = (distributed.get_rank(), distributed.get_world_size()) if distributed else (0, 1)
    if world > 1:
        hosts = [None] * world
        distributed.all_gather_object(hosts, socket.gethostname())
        if len(set(hosts)) != 1:
            raise ValueError('Local retrieval requires ranks on one host; use --no-local-retrieval for a shared multi-host workspace')
    # Existing completed runs already have their public files. Let the miner's
    # native input/contract/output checks validate them without recopying tokens.
    packet = [Path(args.output).joinpath('complete.json').exists() if rank == 0 else None]
    if world > 1:
        distributed.broadcast_object_list(packet, src=0)
    if packet[0]:
        return callback(args)
    workspace = Workspace(args) if rank == 0 else None
    local_args = copy.copy(args)
    packet = [None]
    try:
        if rank == 0:
            try:
                packet[0] = dict(paths=workspace.prepare())
            except Exception as error:
                if world == 1:
                    raise
                packet[0] = dict(error=repr(error))
        if world > 1:
            distributed.broadcast_object_list(packet, src=0)
        if 'error' in packet[0]:
            raise RuntimeError(packet[0]['error'])
        local_args.corpus, local_args.output = map(Path, packet[0]['paths'])
        local_args._local_retrieval_active = True
        result = callback(local_args)
        # Scoring is finished; publication needs no CUDA allocations. The
        # controller may now return this GPU to its teacher while copy-back runs.
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if world > 1:
            distributed.barrier()
        if rank == 0:
            try:
                workspace.publish()
                packet[0] = dict(published=True)
            except Exception as error:
                if world == 1:
                    raise
                packet[0] = dict(error=repr(error))
        if world > 1:
            distributed.broadcast_object_list(packet, src=0)
        if 'error' in packet[0]:
            raise RuntimeError(packet[0]['error'])
        return result
    except BaseException as error:
        if rank == 0 and workspace.owns_lock:
            workspace.status('failed', error=repr(error), scratch_retained=workspace.work.exists())
        raise
    finally:
        if workspace is not None:
            workspace.close()
