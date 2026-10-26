import copy
import os

import pytest
import torch

from colbert_experiments.ordinal_ranking import cached_ranking_backward
from colbert_experiments.retained_ranking import ActivationStorage, retained_ranking_backward


class Student(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4, device=device)
        self.output = torch.nn.Linear(4, 1, device=device)
        self.calls = 0

    def forward(self, query, documents):
        self.calls += 1
        x = torch.tensor([[float(query), float(d)] for d in documents], device=self.linear.weight.device)
        return self.output(self.linear(x).tanh()).flatten()


def batches():
    return [(dict(query=str(i + 1), query_id=str(i), direction='forward' if i % 2 else 'reverse',
                  labels=[0, 5, 2], documents=[dict(text=str(j + 1)) for j in range(3)]), [0, 1, 2])
            for i in range(3)]


def test_retained_forward_count_and_gradient_match_replay():
    torch.manual_seed(3)
    replay = Student()
    retained = copy.deepcopy(replay)
    options = dict(device=torch.device('cpu'), rank=0, world=1, seed_for_micro=lambda i: i)
    expected = cached_ranking_backward(replay, replay, batches(), **options)
    actual = retained_ranking_backward(retained, retained, batches(), **options)
    torch.testing.assert_close(actual['loss'], expected['loss'])
    for a, b in zip(replay.parameters(), retained.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert replay.calls == 6 and retained.calls == 3
    assert actual['encoder_forward_passes'] == actual['model_backward_calls'] == 1
    assert 'replay_backward_seconds' not in actual


def test_pause_during_retained_forward_never_backpropagates_partial_batch():
    model = Student()
    polls = iter([False, True])
    result = retained_ranking_backward(model, model, batches(), device=torch.device('cpu'),
        rank=0, world=1, seed_for_micro=lambda i: i, pause_poll_microbatches=1,
        should_pause=lambda: next(polls))
    assert result is None and model.calls == 1
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.skipif(os.environ.get('MMAI_TEST_RETAINED_CUDA') != '1', reason='Explicit bounded CUDA validation')
@pytest.mark.parametrize('mode,budget', [('gpu', 0), ('auto', 0), ('auto', .000001), ('cpu', 0)])
def test_cuda_activation_storage_preserves_gradients_and_forced_spill(mode, budget):
    torch.manual_seed(4)
    model = Student('cuda')
    reference = copy.deepcopy(model)
    storage = ActivationStorage(model, 'cuda', mode=mode, gpu_budget_gb=budget)
    with storage.context():
        scores = model('2', ['1', '3', '4'])
        loss = scores.square().sum()
    loss.backward()
    expected = reference('2', ['1', '3', '4']).square().sum()
    expected.backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(loss, expected)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    spilled = mode == 'cpu' or mode == 'auto' and budget > 0
    assert bool(storage.offloaded_bytes) == spilled


@pytest.mark.skipif(os.environ.get('MMAI_TEST_RETAINED_CUDA') != '1', reason='Explicit bounded CUDA validation')
def test_host_memory_pressure_stops_before_pinning_more_activations(monkeypatch):
    from colbert_experiments import retained_ranking
    monkeypatch.setattr(retained_ranking, 'available_host_bytes', lambda: 0)
    model = Student('cuda')
    storage = ActivationStorage(model, 'cuda', mode='cpu')
    with pytest.raises(MemoryError, match='will not silently replay'):
        with storage.context():
            x = torch.randn(4, device='cuda', requires_grad=True)
            x.square()
    assert storage.offloaded_bytes == 0


@pytest.mark.skipif(os.environ.get('MMAI_TEST_RETAINED_CUDA') != '1', reason='Explicit bounded CUDA validation')
def test_repeated_saved_tensor_reuses_host_copy_without_retaining_gpu_source():
    import weakref
    model = Student('cuda')
    storage = ActivationStorage(model, 'cuda', mode='cpu')
    tensor = torch.randn(8, device='cuda', requires_grad=True)
    with storage.context():
        loss = (tensor * tensor).sum()
    assert storage.offloaded_tensors == 1
    loss.backward()
    torch.testing.assert_close(tensor.grad, 2 * tensor.detach())
    reference = weakref.ref(tensor)
    del tensor, loss
    assert reference() is None


@pytest.mark.skipif(os.environ.get('MMAI_TEST_RETAINED_CUDA') != '1', reason='Explicit bounded CUDA validation')
def test_host_copy_cache_distinguishes_versions_and_accepts_inference_buffers():
    model = Student('cuda')
    storage = ActivationStorage(model, 'cuda', mode='cpu')
    tensor = torch.ones(4, device='cuda')
    original = storage.pack(tensor)
    tensor.add_(1)
    changed = storage.pack(tensor)
    with torch.inference_mode():
        buffer = torch.full((4,), 3., device='cuda')
    inferred = storage.pack(buffer)
    torch.testing.assert_close(storage.unpack(original), torch.ones(4, device='cuda'))
    torch.testing.assert_close(storage.unpack(changed), torch.full((4,), 2., device='cuda'))
    torch.testing.assert_close(storage.unpack(inferred), torch.full((4,), 3., device='cuda'))
