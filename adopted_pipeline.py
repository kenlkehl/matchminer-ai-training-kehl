"""Monitor an explicitly registered, already-running upstream pipeline."""
import json
from pathlib import Path

from colbert_experiments.local_teacher import process_identity
from trial_cohort_format import file_stamp


class AdoptedPipeline:
    def __init__(self, registration):
        self.expected = registration['process']
        self.pid = self.expected['pid']
        self.root = Path(registration['pipeline_dir'])
        self.stages = registration['stages']
        self.returncode = None
        current = process_identity(self.pid)
        if current is not None and current != self.expected:
            raise ValueError('Adopted pipeline process identity changed')

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        current = process_identity(self.pid)
        if current is not None:
            if current != self.expected:
                raise ValueError('Adopted pipeline PID was reused')
            status_path = self.root/'status.json'
            if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'failed':
                raise RuntimeError('Adopted pipeline reported failure')
            return None
        self.returncode = 1
        status = self.root/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'completed':
            return self.returncode
        for stage in self.stages:
            request, receipt = self.root/'requests'/f'{stage}.json', self.root/'completed'/f'{stage}.json'
            if not request.exists() or not receipt.exists():
                return self.returncode
            wanted, saved = json.loads(request.read_text()), json.loads(receipt.read_text())
            if wanted['fingerprint'] != saved['fingerprint']:
                return self.returncode
            for record in saved['outputs']:
                path = Path(record['path'])
                if not path.exists() or file_stamp(path) != record['stamp']:
                    return self.returncode
        self.returncode = 0
        return self.returncode


class AdoptedArtifactStage:
    """Join a registered miner without relaunching it or trusting its exit alone."""
    def __init__(self, registration, command):
        self.expected = registration['process']
        self.pid = self.expected['pid']
        self.marker = Path(registration['completion_marker'])
        self.required = set(registration['required_outputs'])
        self.returncode = None
        if self.expected['command'] != command or not self.required:
            raise ValueError('Adopted artifact stage command/outputs do not match')
        current = process_identity(self.pid)
        if current is not None and current != self.expected:
            raise ValueError('Adopted artifact stage process identity changed')

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        current = process_identity(self.pid)
        if current is not None:
            if current != self.expected:
                raise ValueError('Adopted artifact stage PID was reused')
            return None
        self.returncode = 1
        if not self.marker.exists():
            return self.returncode
        outputs = json.loads(self.marker.read_text()).get('outputs', {})
        if not self.required.issubset(outputs):
            return self.returncode
        for name in self.required:
            path = self.marker.parent/name
            if not path.is_file() or file_stamp(path) != outputs[name]:
                return self.returncode
        self.returncode = 0
        return self.returncode


class AdoptedLabelStage(AdoptedArtifactStage):
    """Keep an active labeler and require verified final publication on exit."""
    def poll(self):
        if self.returncode is not None:
            return self.returncode
        current = process_identity(self.pid)
        if current is not None:
            if current != self.expected:
                raise ValueError('Adopted labeler PID was reused')
            return None
        self.returncode = 1
        if not self.marker.exists():
            return self.returncode
        saved = json.loads(self.marker.read_text())
        if not saved.get('complete') or not self.required.issubset(saved.get('products', {})):
            return self.returncode
        for name in self.required:
            path = self.marker.parent/name
            if not path.is_file() or file_stamp(path) != saved['products'][name]['stamp']:
                return self.returncode
        self.returncode = 0
        return self.returncode
