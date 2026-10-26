import copy
import gc
import os
import weakref

import pytest
import torch

from colbert_experiments.coherent_activations import pinned_cuda_view, require_coherent_host_access
from colbert_experiments.retained_ranking import ActivationStorage, retained_ranking_backward
from colbert_experiments.ordinal_ranking import cached_ranking_backward
from colbert_experiments.tests.test_retained_ranking import Student, batches


def test_direct_storage_rejects_cpu():
    with pytest.raises(ValueError, match='coherent CUDA'):
        ActivationStorage(Student(), 'cpu', mode='direct')


def test_capability_guard_rejects_non_ats_device(monkeypatch):
    from colbert_experiments import coherent_activations

    class Driver:
        def __init__(self):
            self.cuInit = lambda _: 0
            self.cuDeviceGetAttribute = lambda out, attribute, device: 0

    monkeypatch.setattr(coherent_activations.ctypes, 'CDLL', lambda _: Driver())
    require_coherent_host_access.cache_clear()
    with pytest.raises(RuntimeError, match='coherent ATS'):
        require_coherent_host_access(0)
    require_coherent_host_access.cache_clear()


def test_host_guard_excludes_memory_only_gpu_nodes(tmp_path, monkeypatch):
    from colbert_experiments import retained_ranking
    original_path = retained_ranking.Path
    (tmp_path / 'meminfo').write_text('MemAvailable: 100000000 kB\n')
    nodes = tmp_path / 'nodes'
    for index, cpus, free in [(0, '0-71', 40000000), (1, '', 60000000)]:
        node = nodes / f'node{index}'
        node.mkdir(parents=True)
        (node / 'cpulist').write_text(cpus + '\n')
        (node / 'meminfo').write_text(f'Node {index} MemFree: {free} kB\n')
    monkeypatch.setattr(retained_ranking, 'Path', lambda path: {
        '/proc/meminfo': tmp_path / 'meminfo', '/sys/devices/system/node': nodes}.get(path, original_path(path)))
    assert retained_ranking.available_host_bytes() == 40000000 * 1024
    (tmp_path / 'meminfo').write_text('MemAvailable: 30000000 kB\n')
    assert retained_ranking.available_host_bytes() == 30000000 * 1024


coherent_cuda = pytest.mark.skipif(os.environ.get('MMAI_TEST_COHERENT_CUDA') != '1',
                                  reason='Explicit ATS GPU validation')


@coherent_cuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_direct_view_has_no_hbm_allocation_and_keeps_owner_alive(dtype):
    host = torch.ones((32, 64), dtype=dtype, pin_memory=True)
    owner = weakref.ref(host)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    view = pinned_cuda_view(host, torch.device('cuda', torch.cuda.current_device()))
    assert torch.cuda.memory_allocated() == before
    assert view.data_ptr() == host.data_ptr()
    torch.testing.assert_close((view + view).cpu(), host + host)
    del host
    gc.collect()
    assert owner() is not None
    torch.cuda.synchronize()
    del view
    gc.collect()
    assert owner() is None


@coherent_cuda
def test_direct_retained_gradients_match_replay_on_nondefault_stream():
    torch.manual_seed(4)
    device = torch.device('cuda', torch.cuda.current_device())
    model = Student(device)
    reference = copy.deepcopy(model)
    options = dict(device=device, rank=0, world=1, seed_for_micro=lambda i: 42 + i)
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        actual = retained_ranking_backward(model, model, batches(), **options,
            activation_storage='direct', activation_gpu_budget_gb=.000001)
    expected = cached_ranking_backward(reference, reference, batches(), **options)
    torch.cuda.synchronize()
    assert actual['activation_offloaded_gb'] > 0
    assert actual['encoder_forward_passes'] == 1
    torch.testing.assert_close(actual['loss'], expected['loss'])
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@coherent_cuda
def test_direct_storage_preserves_noncontiguous_saves_and_versions():
    device = torch.device('cuda', torch.cuda.current_device())
    model = Student(device)
    storage = ActivationStorage(model, device, mode='direct', gpu_budget_gb=.000001)
    tensor = torch.arange(24., device=device).reshape(4, 6).T
    original = storage.pack(tensor)
    tensor.add_(7)
    changed = storage.pack(tensor)
    torch.testing.assert_close(storage.unpack(original), torch.arange(24., device=device).reshape(4, 6).T)
    torch.testing.assert_close(storage.unpack(changed), tensor)
    torch.cuda.synchronize()
    assert len(storage.direct_hosts) == 2


@coherent_cuda
@pytest.mark.parametrize('mode', ['cpu', 'direct'])
def test_offload_preserves_attention_alignment_and_broadcast_strides(mode):
    device = torch.device('cuda', torch.cuda.current_device())
    model = Student(device)
    storage = ActivationStorage(model, device, mode=mode, gpu_budget_gb=.000001)
    mask = torch.randn(2, 1, 7, 8, device=device)[..., :7]
    broadcast = torch.arange(6., device=device)[None].expand(5, 6)
    for tensor in (mask, broadcast, mask[..., 1:]):
        restored = storage.unpack(storage.pack(tensor))
        assert restored.stride() == tensor.stride()
        torch.testing.assert_close(restored, tensor)
    torch.cuda.synchronize()
