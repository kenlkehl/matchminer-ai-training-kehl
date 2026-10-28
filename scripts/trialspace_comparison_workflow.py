"""Queued notes → two exhaustive retrievers → shared labels → local dense training.

The configuration contains exact commands and process identities. Generation is
only observed. Cloud teachers stop at inference completion; iterative dense
rounds can use the local/LAN pool while ColBERT labeling remains deferred.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gcp_note_fleet import now, save, process_identity, Fleet
from scripts.post_notes_workflow import generation_ready, busy_gpus, healthy_teacher
from colbert_experiments.local_teacher import process_identity as identity, open_pidfd, send_pidfd_signal
from trial_cohort_format import file_stamp


def validate_teacher(target, current, environment):
    if current != target['process']:
        raise ValueError('Registered teacher process identity changed')
    command = current['command']
    port = command[command.index('--port') + 1] if '--port' in command else '8000'
    if ('serve' not in command or target['model_root'] not in command
            or port != str(target['port']) or environment.get('CUDA_VISIBLE_DEVICES') != str(target['gpu'])):
        raise ValueError('Registered teacher command/GPU/port does not match')


def teacher_environment(target):
    env = {**os.environ, **target.get('environment', {}), 'CUDA_VISIBLE_DEVICES': str(target['gpu'])}
    executable = Path(target['command'][0])
    if executable.is_absolute():
        # Keep the virtualenv path (not the resolved Python symlink). vLLM's
        # kernel builders invoke tools such as ninja through PATH.
        env['PATH'] = str(executable.parent) + os.pathsep + env.get('PATH', os.defpath)
    return env


def cloud_states(config):
    """Live GCP verification; a historical shutdown receipt is insufficient."""
    def one(project):
        result = subprocess.run([config['gcloud'], 'compute', 'instances', 'list', '--project', project,
            '--filter=name~rtx', '--format=json(name,status)'], text=True, capture_output=True,
            check=True, timeout=60)
        return {(project, row['name']): row['status'] for row in json.loads(result.stdout)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = dict(item for group in pool.map(one, sorted({v['project'] for v in config['vms']}))
                    for item in group.items())
    expected = [(v['project'], v['name']) for v in config['vms']]
    if len(expected) != 8 or any(key not in rows for key in expected):
        raise ValueError('Cannot verify the full eight-VM allowlist')
    return {project + '/' + name: rows[project, name] for project, name in expected}


def require_cloud_stopped(config):
    states = cloud_states(config)
    if any(state != 'TERMINATED' for state in states.values()):
        raise RuntimeError('Training requires all eight cloud VMs to be TERMINATED: ' + repr(states))
    return states


class Workflow:
    def __init__(self, config, directory):
        preparation = config.get('stages', {}).get('preparing_corpus', {}).get('command')
        if preparation and ('--merge-source-splits' not in preparation or '--no-merge-source-splits' in preparation):
            raise ValueError('Production comparison must merge all synthetic source splits')
        if config.get('dense_round_two') and not config.get('train_dense_before_colbert_labels'):
            raise ValueError('Round two requires dense-first training')
        self.config, self.root = config, Path(directory)
        self.status = dict(pid=os.getpid(), started_at=now(), phase='waiting_notes', child_pid=None)
        self.children = {}
        self.cancelled = False
        self.summary_started_at = None
        (self.root / 'logs').mkdir(parents=True, exist_ok=True)
        self.teacher = dict(config['teacher'])
        self.teacher_process = None
        saved = self.root / 'teacher_process.json'
        if saved.exists():
            self.teacher['process'] = json.loads(saved.read_text())

    def update(self, phase=None, **values):
        self.status.update(values, updated_at=now())
        if phase:
            self.status['phase'] = phase
        save(self.root / 'status.json', self.status)

    def check(self):
        if self.cancelled or (self.root / 'STOP').exists():
            raise RuntimeError('Comparison workflow stopped by operator')
        if 'labeling_dense_round2' in self.children:
            self.sync_round_two_registry()
        for name, child in self.children.items():
            if child.poll() not in (None, 0):
                raise RuntimeError(f'{name} exited with code {child.returncode}; see logs/{name}.log')

    def wait_notes(self):
        while True:
            self.check()
            status = json.loads(Path(self.config['notes_status']).read_text())
            alive = process_identity(self.config['generator_pid']) == self.config['generator_start_ticks']
            if generation_ready(status, alive):
                return
            self.update('waiting_notes', saved_notes=status.get('stage_completed'), expected_notes=status.get('expected_notes'))
            time.sleep(10)

    def receipt(self, name):
        return self.root / 'completed' / (name + '.json')

    def done(self, name):
        path, stage = self.receipt(name), self.config['stages'][name]
        if not path.exists():
            return False
        prior = json.loads(path.read_text())
        if prior['command'] != stage['command'] or any(
                file_stamp(p) != stamp for p, stamp in prior['outputs'].items()):
            raise ValueError('Completed workflow stage changed: ' + name)
        return True

    def start(self, name):
        dense_first = self.config.get('train_dense_before_colbert_labels', False)
        if name == 'labeling' and self.config.get('dense_round_two'):
            raise ValueError('ColBERT labeling is deferred by the EmbeddingGemma round-two plan')
        if name in ('mining_dense_round2', 'labeling_dense_round2'):
            from embeddinggemma_round_two import validate_stage
            validate_stage(self.config, name)
            if not (self.done('training') and self.done('evaluation')):
                raise ValueError('Round two requires completed round-one training and diagnostics')
            require_cloud_stopped(json.loads(Path(self.config['fleet_config']).read_text()))
            if name == 'labeling_dense_round2':
                if not self.done('mining_dense_round2'):
                    raise ValueError('Round-two labeling requires verified round-two retrieval')
                self.sync_round_two_registry()
        if name == 'labeling' and dense_first and not (self.done('training') and self.done('evaluation')):
            raise ValueError('ColBERT labeling requires completed EmbeddingGemma training and diagnostics')
        if name == 'training' and self.config.get('training_candidates_file'):
            selection = json.loads(Path(self.config['training_candidates_file']).read_text())['source']
            if selection not in ('union', 'embeddinggemma2'):
                raise ValueError('Unknown EmbeddingGemma training candidate source')
            if dense_first and selection != 'embeddinggemma2':
                raise ValueError('Training before ColBERT labeling requires EmbeddingGemma-only candidates')
            command = self.config['stages'][name]['command']
            labels_root = Path(self.config.get('training_pair_labels_root', self.root / 'pair_labels'))
            command[command.index('--input-parquet') + 1] = str(labels_root / selection / 'pairs.parquet')
        if self.done(name):
            return
        if name == 'labeling_dense' and self.config.get('resume_fleet_for_labeling'):
            self.resume_fleet_for_labeling()
        stage = self.config['stages'][name]
        adopted = self.config.get('adopt_stages', {}).get(name)
        if adopted:
            from adopted_pipeline import AdoptedPipeline, AdoptedArtifactStage, AdoptedLabelStage
            child = (AdoptedLabelStage(adopted, stage['command']) if adopted.get('kind') == 'labels' else
                     AdoptedArtifactStage(adopted, stage['command'])
                     if adopted.get('kind') == 'artifacts' else AdoptedPipeline(adopted))
            self.children[name] = child
            self.update(name, child_pid=child.pid, command=stage['command'], adopted=True)
            if name == 'summarizing':
                self.summary_started_at = time.time()
            return
        self.update(name, child_pid=None)
        env = {**os.environ, **self.config.get('environment', {}), **stage.get('environment', {})}
        with (self.root / 'logs' / (name + '.log')).open('ab') as stream:
            child = subprocess.Popen(stage['command'], cwd=self.config['repo'], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        if name == 'summarizing':
            self.summary_started_at = time.time()
        save(self.root / 'processes' / (name + '.json'), identity(child.pid))
        self.update(child_pid=child.pid, command=stage['command'])

    def wait(self, name):
        if name not in self.children:
            return
        child = self.children[name]
        while child.poll() is None:
            self.check()
            self.update()
            time.sleep(10)
        self.check()
        stage = self.config['stages'][name]
        save(self.receipt(name), dict(at=now(), command=stage['command'],
             outputs={p: file_stamp(p) for p in stage['outputs']}))
        del self.children[name]
        self.update(child_pid=None)

    def stage(self, name):
        self.start(name)
        self.wait(name)

    def summaries_for_retrieval(self):
        if not self.config.get('overlap_summary_publication', False):
            self.stage('summarizing')
            return
        from summary_overlap import paths_for, local_products_ready, release_products
        self.start('summarizing')
        if 'summarizing' not in self.children:
            return
        paths = paths_for(self.config['stages']['summarizing']['command'])
        while self.children['summarizing'].poll() is None:
            self.check()
            if local_products_ready(paths, started_at=self.summary_started_at):
                released = release_products(paths, self.root/'summary_retrieval_ready.json')
                if released:
                    self.update('summaries_ready', summary_publication_pending=True,
                                summary_patients=released['patients'])
                    return
            self.update()
            time.sleep(5)
        self.wait('summarizing')

    def join_summary_publication(self):
        self.wait('summarizing')
        from summary_overlap import verify_published_products
        verify_published_products(self.root/'summary_retrieval_ready.json')
        self.update(summary_publication_pending=False)

    def registry_teacher(self, enabled):
        registry = Path(self.config['registry'])
        target = self.config['teacher']['server_entry']
        with registry.with_name('servers.fleet.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(registry.read_text())
            rows = [s for s in value['servers'] if s['url'] != target['url']]
            if enabled:
                rows.append(target)
            save(registry, dict(value, servers=rows, ready=bool(rows), updated_at=now()))

    def release_teacher(self):
        self.verify_gpu()
        self.update('releasing_gb300', child_pid=None)
        self.registry_teacher(False)
        expected = self.teacher['process']
        current = identity(expected['pid'])
        if current is not None:
            fd = open_pidfd(expected['pid'])
            try:
                env = dict(part.split('=', 1) for part in Path(f'/proc/{expected["pid"]}/environ')
                           .read_bytes().decode().split('\0') if '=' in part)
                validate_teacher(self.teacher, identity(expected['pid']), env)
                send_pidfd_signal(fd, signal.SIGTERM)
            finally:
                os.close(fd)
        save(self.root / 'last_teacher_release.json', dict(at=now(), process=expected))
        deadline = time.monotonic() + 300
        while busy := busy_gpus([str(self.teacher['gpu'])]):
            self.check()
            if self.teacher_process is not None:
                self.teacher_process.poll()
            if time.monotonic() > deadline:
                raise RuntimeError('GB300 remains occupied after registered teacher stopped: ' + repr(busy))
            time.sleep(5)

    def adopting_live_miner(self):
        """Only the exact registered miner can retain the GB300 on restart."""
        registration = self.config.get('adopt_stages', {}).get('mining_colbert')
        if not registration or registration.get('kind') != 'artifacts':
            return False
        from adopted_pipeline import AdoptedArtifactStage
        child = AdoptedArtifactStage(registration, self.config['stages']['mining_colbert']['command'])
        if child.poll() is not None:
            return False
        self.verify_gpu()
        environment = dict(part.split('=', 1) for part in Path(f'/proc/{child.pid}/environ')
                           .read_bytes().decode().split('\0') if '=' in part)
        if environment.get('CUDA_VISIBLE_DEVICES') != str(self.teacher['gpu']):
            raise ValueError('Adopted miner is not assigned to the registered GB300')
        if any(row['pid'] != child.pid for row in busy_gpus([str(self.teacher['gpu'])])):
            raise ValueError('An unregistered process occupies the GB300 during miner adoption')
        if identity(child.pid) != child.expected:
            raise ValueError('Adopted miner changed while checking its GPU')
        self.registry_teacher(False)
        return True

    def restore_teacher(self, publishing_process=None):
        self.verify_gpu()
        self.update('restoring_gb300', child_pid=None)
        if identity(self.teacher['process']['pid']) != self.teacher['process']:
            busy = busy_gpus([str(self.teacher['gpu'])])
            allowed = (publishing_process and identity(publishing_process['pid']) == publishing_process)
            if any(not allowed or row['pid'] != publishing_process['pid'] for row in busy):
                raise RuntimeError('Refusing to start a teacher on an occupied GB300')
            env = teacher_environment(self.teacher)
            with (self.root / 'logs/gb300_teacher.log').open('ab') as stream:
                self.teacher_process = subprocess.Popen(self.teacher['command'], env=env, cwd=self.config['repo'],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.teacher['process'] = identity(self.teacher_process.pid)
            save(self.root / 'teacher_process.json', self.teacher['process'])
        deadline = time.monotonic() + 1800
        while not healthy_teacher(self.teacher):
            self.check()
            if identity(self.teacher['process']['pid']) != self.teacher['process']:
                raise RuntimeError('GB300 teacher exited while starting')
            if time.monotonic() > deadline:
                raise TimeoutError('GB300 teacher failed health/model checks')
            time.sleep(10)
        self.registry_teacher(True)

    def mine_colbert_and_restore_teacher(self):
        """Return the GPU to labeling while verified retrieval publication runs."""
        self.start('mining_colbert')
        child = self.children.get('mining_colbert')
        restored = False
        while child is not None and child.poll() is None:
            self.check()
            from retrieval_publication import publication_ready
            expected = getattr(child, 'expected', None) or identity(child.pid)
            proof = publication_ready(self.config['stages']['mining_colbert']['command'], expected) if expected else None
            if proof:
                command = self.teacher['command']
                fraction = float(command[command.index('--gpu-memory-utilization')+1]) if '--gpu-memory-utilization' in command else .9
                total, free = map(float, subprocess.check_output(['nvidia-smi', '--id='+str(self.teacher['gpu']),
                    '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],text=True).strip().split(','))
                if free >= total*fraction:
                    save(self.root/'retrieval_publication_overlap.json',dict(at=now(),**proof))
                    self.update('restoring_gb300', retrieval_publication_pending=True)
                    self.restore_teacher(publishing_process=expected)
                    restored = True
                    self.update('publishing_colbert', retrieval_publication_pending=True)
                    break
            self.update()
            time.sleep(5)
        self.wait('mining_colbert')
        self.update(retrieval_publication_pending=False)
        if not restored:
            self.restore_teacher()

    def verify_gpu(self):
        actual = subprocess.check_output(['nvidia-smi', '--id=' + str(self.teacher['gpu']),
            '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if actual != self.teacher['gpu_uuid']:
            raise ValueError('The selected GPU is not the registered GB300')

    def stop_cloud(self):
        self.update('stopping_cloud', child_pid=None)
        config_path = Path(self.config['fleet_config'])
        fleet = json.loads(config_path.read_text())
        (config_path.parent / 'STOP').touch()
        while True:
            self.check()
            states = cloud_states(fleet)
            report = json.loads((config_path.parent / 'status.json').read_text())
            alive = process_identity(report['supervisor_pid']) is not None
            complete = config_path.parent / 'shutdown_complete.json'
            # The controller joins pending Spot-start commands before declaring
            # completion. Live verification then protects the training boundary.
            if complete.exists() and json.loads(complete.read_text()).get('all_terminated'):
                if all(s == 'TERMINATED' for s in states.values()):
                    save(self.root / 'cloud_shutdown_verified.json', dict(at=now(), states=states,
                         stop_mtime_ns=(config_path.parent/'STOP').stat().st_mtime_ns))
                    return
            if not alive:
                Fleet(fleet, config_path.parent).shutdown('comparison_pretraining_cleanup')
            time.sleep(10)

    def resume_fleet_for_labeling(self, *, after_training=False):
        """Honor an explicit pause/restart request only when corrected candidates are ready."""
        if after_training and self.config.get('dense_round_two'):
            raise ValueError('Cloud restart is disabled by the local/LAN round-two plan')
        request = self.config['resume_fleet_for_labeling']
        if after_training and not (self.done('training') and self.done('evaluation')):
            raise ValueError('Cannot restart the ColBERT fleet before EmbeddingGemma training and diagnostics finish')
        receipt = self.root/('fleet_restart_after_training.json' if after_training else 'fleet_restart_requested.json')
        if receipt.exists():
            return
        if not self.done('mining_dense'):
            raise ValueError('Cannot restart the fleet before corrected dense retrieval finishes')
        self.update('starting_label_fleet', child_pid=None)
        pause_path = self.root/'cloud_shutdown_verified.json' if after_training else Path(request['pause_receipt'])
        pause = json.loads(pause_path.read_text())
        config_path = Path(self.config['fleet_config'])
        stop = config_path.parent/'STOP'
        if not stop.exists() or stop.stat().st_mtime_ns != pause['stop_mtime_ns']:
            raise ValueError('Fleet STOP changed since the explicit pause/restart request')
        active = subprocess.check_output(['systemctl','--user','show',request['unit'],'-p','MainPID','--value'],text=True).strip()
        if int(active or 0):
            raise RuntimeError('Previous fleet supervisor has not finished shutting down')
        fleet = json.loads(config_path.read_text())
        current = identity(os.getpid())
        fleet['workflow'] = dict(fleet.get('workflow', {}), status=str(self.root/'status.json'),pid=current['pid'],
            start_ticks=str(current['start_ticks']),label_status=str(self.root/'labels_union/status.json'))
        save(config_path,fleet)
        shutdown = config_path.parent/'shutdown_complete.json'
        if shutdown.exists():
            save(self.root/'fleet_prior_shutdown.json', json.loads(shutdown.read_text()))
            shutdown.unlink()
        stop.unlink()
        loaded = subprocess.check_output(['systemctl','--user','show',request['unit'],'-p','LoadState','--value'],text=True).strip()
        if loaded == 'not-found':
            subprocess.run(['systemd-run','--user','--unit='+request['unit'],
                '--property=Restart=on-failure','--property=RestartSec=30',
                '--property=KillMode=control-group','--property=TimeoutStopSec=180',
                '--property=StartLimitIntervalSec=0','--property=LimitNOFILE=65536',
                '--property=StandardOutput=append:'+str(config_path.parent/'supervisor.log'),
                '--property=StandardError=append:'+str(config_path.parent/'supervisor.log'),
                *request['command']],check=True)
        else:
            subprocess.run(['systemctl','--user','start',request['unit']],check=True)
        pid = int(subprocess.check_output(['systemctl','--user','show',request['unit'],'-p','MainPID','--value'],text=True).strip())
        process = identity(pid)
        if process is None:
            raise RuntimeError('Fleet supervisor did not start')
        launch = json.loads((config_path.parent/'launch.json').read_text())
        launch.update(pid=pid,start_ticks=str(process['start_ticks']),command=process['command'],
                      workflow=fleet['workflow'],started_at=now())
        save(config_path.parent/'launch.json',launch)
        save(receipt,dict(at=now(),process=process,pause=pause,corrected_dense_candidates_ready=True,
                         after_embeddinggemma_training=after_training))

    def train_dense_first(self):
        """Fit and assess only dense candidates before the ColBERT label stage."""
        needs_training = not (self.done('training') and self.done('evaluation'))
        if needs_training:
            self.stop_cloud()
        self.stage('project_dense_labels')
        if self.config.get('local_dense_training'):
            self.start('publish_dense_training_pairs')
        if needs_training:
            self.release_teacher()
            require_cloud_stopped(json.loads(Path(self.config['fleet_config']).read_text()))
            self.stage('training')
            # Background archive and pair publication must finish before later
            # stages consume shared artifacts or import old judgments.
            self.wait('labeling_dense')
            self.wait('publish_dense_training_pairs')
            self.stage('evaluation')
        if self.config.get('local_dense_training'):
            self.wait('labeling_dense')
            self.wait('publish_dense_training_pairs')
            self.update(label_publication_pending=False)
        if not self.config.get('dense_round_two'):
            self.restore_teacher()
            self.resume_fleet_for_labeling(after_training=True)

    def wait_dense_training_scores(self):
        if not self.config.get('local_dense_training'):
            return self.wait('labeling_dense')
        from dense_training_preparation import score_source
        while 'labeling_dense' in self.children:
            self.check()
            if score_source(self.config['local_dense_training']) is not None:
                self.update('dense_scores_ready', child_pid=None, label_publication_pending=True)
                return
            if self.children['labeling_dense'].poll() is not None:
                return self.wait('labeling_dense')
            self.update('waiting_dense_scores')
            time.sleep(10)

    def dense_labels_need_teacher(self):
        """A controller restart during export/training must not reload vLLM."""
        if self.done('labeling_dense'):
            return False
        if self.config.get('local_dense_training'):
            from dense_training_preparation import score_source
            if score_source(self.config['local_dense_training']) is not None:
                return False
        return True

    def sync_round_two_registry(self):
        from embeddinggemma_round_two import sync_registry
        sync_registry(self.config)

    def run_dense_round_two(self):
        """Re-retrieve with round one's student, then label on local/LAN only."""
        plan = self.config['dense_round_two']
        self.stop_cloud()
        if not self.done('mining_dense_round2'):
            self.release_teacher()
        self.stage('mining_dense_round2')
        self.restore_teacher()
        self.sync_round_two_registry()
        save(Path(plan['labels'])/'reuse_sources.json', {'sources': [dict(
            self.config['label_reuse'], mode='candidate_intersection',
            corpus=plan['corpus'])]})
        self.stage('labeling_dense_round2')
        self.stage('project_dense_round2')
        require_cloud_stopped(json.loads(Path(self.config['fleet_config']).read_text()))
        self.update('awaiting_round_two_review', completed_at=now(), child_pid=None,
                    colbert_labeling_deferred=True, cloud_enabled=False,
                    round_two_labels=plan['labels'], round_two_pairs=plan['pairs'])

    def run(self):
        def halt(signum, frame):
            self.cancelled = True
        signal.signal(signal.SIGTERM, halt)
        signal.signal(signal.SIGINT, halt)
        try:
            self.update()
            self.wait_notes()
            self.summaries_for_retrieval()
            self.stage('preparing_corpus')
            if not (self.done('mining_dense') and self.done('mining_colbert')):
                if not self.adopting_live_miner():
                    self.release_teacher()
                self.stage('mining_dense')
                # These labels are imported by the final union labeler. Each pair
                # is judged once even while ColBERT retrieval is still running.
                self.start('labeling_dense')
                self.mine_colbert_and_restore_teacher()
                self.update('waiting_dense_labels', child_pid=None)
                self.wait_dense_training_scores()
            else:
                self.start('labeling_dense')
                if self.dense_labels_need_teacher():
                    self.restore_teacher()
                self.update('waiting_dense_labels', child_pid=None)
                self.wait_dense_training_scores()
            if self.config.get('overlap_summary_publication', False):
                self.join_summary_publication()
            dense_first = self.config.get('train_dense_before_colbert_labels', False)
            if dense_first:
                self.train_dense_first()
            if self.config.get('dense_round_two'):
                self.run_dense_round_two()
                return
            self.stage('union')
            labels = Path(self.config['union_labels'])
            save(labels / 'reuse_sources.json', {'sources': [self.config['label_reuse']]})
            self.stage('labeling')
            self.stage('project_labels')
            self.stage('report')
            self.stop_cloud()
            if not dense_first:
                self.release_teacher()
                require_cloud_stopped(json.loads(Path(self.config['fleet_config']).read_text()))
                self.stage('training')
                self.stage('evaluation')
                self.restore_teacher()
            self.update('completed', completed_at=now(), child_pid=None)
        except BaseException as error:
            for child in self.children.values():
                try:
                    alive = child.poll() is None
                except Exception:
                    alive = getattr(child, 'expected', None) == identity(child.pid)
                if alive:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            self.update('failed', error=repr(error))
            # The attached fleet joins maintenance, stops its eight-VM allowlist,
            # and publishes a shutdown receipt. Existing LAN servers are untouched.
            (Path(self.config['fleet_config']).parent / 'STOP').touch()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    with (args.config.parent / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Workflow(json.loads(args.config.read_text()), args.config.parent).run()


if __name__ == '__main__':
    main()
