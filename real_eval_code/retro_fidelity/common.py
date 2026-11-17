"""Helpers shared by the retrospective distillation-fidelity stages (see __init__.py).

Every output of this evaluation describes real patients (PHI): directories are created at mode 700 and files at
600 (``private()`` sets umask 077 for the process and its children), and nothing here prints or logs clinical text
or medical record numbers; status lines carry counts only (``mask`` hides any number of 6 or more digits).
"""
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trial_cohort_format import file_stamp  # noqa: E402
from trialspace import stage  # noqa: E402
from trialspace.data import digest, read_jsonl, write_json, write_jsonl  # noqa: E402

# Every patient and cohort of this evaluation is in the corpus split "val" (the labeler requires a named split;
# the patients are the SOC cohort's validation split, a development set).
SPLIT = 'val'
WORKFLOWS = ('patient_centric', 'trial_centric')
DIRECTION = {'patient_centric': 'patient_to_cohort', 'trial_centric': 'cohort_to_patient'}
L = Path('/ksg/kehl_mm_data/mmai/v23/data/no_phi/synthetic_dataset_n5000_screened_20261106')
EMBEDDERS = {
    # (a) base EmbeddingGemma 2, as training's round-1 mining encoded with it
    'embeddinggemma2': dict(model='google/embeddinggemma-2', revision='914f7f89142e33e77833254d9c9b90c3cef7303b',
                            cache_dir='/data1/ken/models/mmai-app-hf'),  # the complete 914f7f89 snapshot on camus
    # (b) the round-2 TrialSpace fit, the inference default
    'trialspace_round2': dict(model=str(L / 'trialspace/models/trialspace_round2'), revision=None, cache_dir=None),
}
DECISION_MODEL = L / 'trialspace/models/decision_model'
RANKERS = ('embeddinggemma2', 'trialspace_round2', 'trialspace_round2_decision')
TOP = 20


def private():
    """umask 077 for this process and every process it starts: new files 600, new directories 700."""
    os.umask(0o077)


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def mask(value):
    """Text safe to print: any run of 6 or more digits becomes #s."""
    return re.sub(r'\d{6,}', lambda m: '#' * len(m.group()), str(value))


def log(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S %Z")}] {mask(message)}', flush=True)


def stamp_inputs(paths):
    return {Path(p).name: file_stamp(p) for p in paths}


def hold_gb300(holders, note):
    """Register this process as holding the OA pause (data/phi/real_time/resume_oa_after_fidelity_20261117.py
    resumes OA once no registered process has been alive for its grace period)."""
    if not holders:
        return
    import fcntl
    holders = Path(holders)
    pid = os.getpid()
    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    with holders.with_name(holders.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(holders.read_text()) if holders.exists() else []
        rows.append(dict(pid=pid, start_ticks=ticks, note=note, at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        temporary = holders.with_name(holders.name + f'.{pid}.tmp')
        temporary.write_text(json.dumps(rows, indent=2) + '\n')
        temporary.replace(holders)


def run_stage(directory, kind, identity, outputs, work, *, check=False, content=None, upstream=None):
    """Stage protocol of the training repository: --check answers 0/10/20; a complete directory is skipped; a
    directory begun under other settings is refused; ``work()`` runs, then the receipt is written last."""
    if check:
        stage.exit_check(directory, kind, identity)
    if stage.require_new_or_same(directory, kind, identity):
        log(f'[skip] {kind} already complete')
        return stage.read_receipt(directory)
    private_dir(directory)
    result = work() or {}
    names = outputs() if callable(outputs) else outputs
    return stage.write_receipt(directory, kind, identity, names, content={**(content or {}), **result},
                               upstream=upstream)


__all__ = ['DECISION_MODEL', 'DIRECTION', 'EMBEDDERS', 'L', 'RANKERS', 'ROOT', 'SPLIT', 'TOP', 'WORKFLOWS', 'digest',
           'file_stamp', 'hold_gb300', 'log', 'mask', 'private', 'private_dir', 'read_jsonl', 'run_stage', 'stage',
           'stamp_inputs', 'write_json', 'write_jsonl']
