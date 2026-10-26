"""Full-batch ranking gradients with one forward per microbatch, retaining graphs."""
from contextlib import nullcontext
import math
from pathlib import Path
import time
import weakref

import torch
import torch.distributed as dist

from .ordinal_ranking import PAUSE_POLL_MICROBATCHES, ranking_loss
from .ranking_microbatches import prepare_encoder_batches
from .coherent_activations import storage_span


def available_host_bytes():
    available = None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
        # In Grace NUMA mode MemAvailable includes GPU HBM. Conservatively
        # bound host storage by free RAM on nodes with CPUs, excluding GPU-only
        # nodes. Do not count that HBM a second time as offload capacity.
        cpu_free = []
        for node in Path('/sys/devices/system/node').glob('node[0-9]*'):
            if (node / 'cpulist').read_text().strip():
                for line in (node / 'meminfo').read_text().splitlines():
                    if 'MemFree:' in line:
                        cpu_free.append(int(line.split()[-2]) * 1024)
        if cpu_free:
            return min(available, sum(cpu_free)) if available is not None else sum(cpu_free)
        return available
    except (OSError, ValueError):
        pass
    return available


class ActivationStorage:
    """Retain saved tensors on GPU or spill to pinned RAM without recomputation.

    Auto/direct's soft limit leaves half of initially free device memory for forward
    temporaries, restored tensors and backward. It is not a hard VRAM limit.
    Parameter/buffer storage remains on device because the model owns it anyway.
    Direct reads spilled pinned memory through CUDA views rather than restoring it.
    """
    def __init__(self, model, device, *, mode='auto', gpu_budget_gb=0):
        if mode not in ('auto', 'gpu', 'cpu', 'direct') or not math.isfinite(gpu_budget_gb) or gpu_budget_gb < 0:
            raise ValueError('Expected auto/gpu/cpu/direct activation storage and nonnegative GPU budget')
        self.mode, self.device = mode, torch.device(device)
        if mode == 'direct':
            from .coherent_activations import require_coherent_host_access
            if self.device.type != 'cuda':
                raise ValueError('Direct activation storage requires a coherent CUDA GPU')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            require_coherent_host_access(self.device.index)
        self.direct_hosts = []
        self.offloaded_bytes = self.offloaded_tensors = 0
        self.next_host_check = 0
        self.host_copies = {}
        self.limit = None
        self.persistent = {t.untyped_storage().data_ptr() for t in
                           [*model.parameters(), *model.buffers()] if t.device.type == 'cuda'}
        if mode in ('auto', 'direct') and self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            # The CUDA allocator can reuse its reserved-but-inactive memory.
            # Excluding it would ratchet the limit down after each large step.
            free += torch.cuda.memory_reserved(self.device) - allocated
            self.limit = (int(gpu_budget_gb * 2**30) if gpu_budget_gb else
                          allocated + int(free * .5))

    def pack(self, tensor):
        key = None if tensor.is_inference() else (id(tensor), tensor._version)
        prior = self.host_copies.get(key)
        if prior is not None and prior[0]() is tensor:
            return prior[1]
        if (tensor.device.type != 'cuda' or self.mode == 'gpu' or not tensor.numel()
                or tensor.untyped_storage().data_ptr() in self.persistent
                or (self.mode in ('auto', 'direct') and torch.cuda.memory_allocated(self.device) < self.limit)):
            return tensor.detach()
        span = storage_span(tensor)
        size = span * tensor.element_size()
        # Preserve physical strides, including SDPA's aligned mask rows and
        # zero-stride broadcast views. Compacting them can make backward choose
        # different kernels or fail alignment checks. Copy the bounded storage
        # span once; no repeated writes into overlapping broadcast locations.
        if self.offloaded_bytes + size >= self.next_host_check:
            available = available_host_bytes()
            # PyTorch rounds pinned allocations to power-of-two blocks.
            allocation = 1 << (max(1, size) - 1).bit_length()
            if available is not None and available < allocation + 8 * 2**30:
                raise MemoryError('Retained activations would leave less than 8 GiB available host RAM. '
                                  'Choose replay or adjust the GPU retention budget; '
                                  'the trainer will not silently replay or discard examples.')
            self.next_host_check = self.offloaded_bytes + size + 2**30
        backing = torch.empty(span, dtype=tensor.dtype, device='cpu', pin_memory=True)
        backing.copy_(tensor.detach().as_strided((span,), (1,)), non_blocking=True)
        host = backing.as_strided(tensor.shape, tensor.stride())
        if self.mode == 'direct':
            # Keep every host allocation alive through the synchronized model
            # backward, including saves of inference tensors absent from cache.
            self.direct_hosts.append(host)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(tensor.device))
        self.offloaded_bytes += size
        self.offloaded_tensors += 1
        saved = tensor.device, host, ready
        # Repeated uses of an unchanged tensor (notably attention masks) share
        # one host copy. Weak identity plus version prevents pointer-reuse and
        # in-place-update collisions without keeping the source tensor on GPU.
        if key is not None:
            self.host_copies[key] = weakref.ref(tensor), saved
        return saved

    def unpack(self, saved):
        if isinstance(saved, torch.Tensor):
            return saved
        device, host, ready = saved
        torch.cuda.current_stream(device).wait_event(ready)
        if self.mode == 'direct':
            from .coherent_activations import pinned_cuda_view
            return pinned_cuda_view(host, device)
        backing = host.as_strided((storage_span(host),), (1,)).to(device, non_blocking=True)
        return backing.as_strided(host.shape, host.stride())

    def context(self):
        return (nullcontext() if self.mode == 'gpu' or self.device.type != 'cuda' else
                torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack))

    def statistics(self):
        return dict(activation_storage=self.mode, activation_offloaded_gb=self.offloaded_bytes / 2**30,
                    activation_offloaded_tensors=self.offloaded_tensors,
                    activation_gpu_soft_limit_gb=None if self.limit is None else self.limit / 2**30)


def retained_ranking_backward(model, wrapped, batch, *, device, rank, world,
                              seed_for_micro, objective='legacy_ranking', scale=20.,
                              contrastive_weight=1., should_pause=lambda: False,
                              pause_poll_microbatches=PAUSE_POLL_MICROBATCHES,
                              groups_per_forward=1, encoder_token_budget=32768,
                              activation_storage='auto', activation_gpu_budget_gb=0,
                              profile_memory=False):
    """Differentiate the same global loss, with retained activations and no replay."""
    if pause_poll_microbatches < 1:
        raise ValueError('Pause polling interval must be positive')
    labels, directions, queries, offsets = [], [], [], [0]
    for group, indices in batch:
        labels.extend(group['labels'][i] for i in indices)
        directions.extend([group['direction']] * len(indices))
        queries.extend([group['query_id']] * len(indices))
        offsets.append(len(labels))
    started = time.monotonic()
    packets, fallback = prepare_encoder_batches(model, batch, offsets, rank=rank, world=world,
        device=device, groups_per_forward=groups_per_forward, token_budget=encoder_token_budget)
    microbatches = len(packets)
    if world > 1:
        count = torch.tensor(microbatches, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        microbatches = int(count)
    preparation_seconds = time.monotonic() - started
    storage = ActivationStorage(model, device, mode=activation_storage, gpu_budget_gb=activation_gpu_budget_gb)
    values = torch.zeros(len(labels), device=device, dtype=torch.float32)
    predictions = []
    memory_metrics = {}
    if profile_memory and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with storage.context():
        for micro in range(microbatches):
            if micro % pause_poll_microbatches == 0 and should_pause():
                return None
            real = micro < len(packets)
            packet = packets[micro] if real else None
            torch.manual_seed(seed_for_micro(packet.seed_micro if real else micro))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                prediction = wrapped(**(packet.kwargs if real else fallback)).float()
            predictions.append(prediction)
            if real:
                values[packet.positions] = prediction.detach()
    if should_pause():
        return None
    if world > 1:
        dist.all_reduce(values)
    # Only scalar scores and their loss derivatives cross to CPU. The model
    # graphs remain alive, so the following backward needs no second forward.
    loss_scores = values.cpu().requires_grad_()
    forward_seconds = time.monotonic() - started
    if profile_memory and device.type == 'cuda':
        memory_metrics.update(forward_peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
                              forward_end_allocated_gb=torch.cuda.memory_allocated(device) / 2**30)
    if not torch.isfinite(loss_scores).all():
        raise ValueError('Nonfinite retained MaxSim score')
    started = time.monotonic()
    if profile_memory and device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    loss, components = ranking_loss(loss_scores, labels, directions, queries,
        objective=objective, scale=scale, contrastive_weight=contrastive_weight)
    score_grad, = torch.autograd.grad(loss, loss_scores)
    score_grad = score_grad.to(device)
    gradients = [score_grad[packets[micro].positions] * world if micro < len(packets)
                 else torch.zeros_like(prediction) for micro, prediction in enumerate(predictions)]
    loss_seconds = time.monotonic() - started
    started = time.monotonic()
    # One autograd invocation traverses all retained model graphs. DDP averages
    # their parameter gradients; the world multiplier above undoes that average.
    try:
        torch.autograd.backward(predictions, gradients)
    finally:
        # Also protect externally owned host views if backward raises after
        # enqueuing kernels. They must outlive all asynchronous GPU reads.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    backward_seconds = time.monotonic() - started
    if profile_memory and device.type == 'cuda':
        memory_metrics['backward_peak_allocated_gb'] = torch.cuda.max_memory_allocated(device) / 2**30
    if should_pause():
        return None
    return dict(loss=loss.detach(), **{k: v.detach() if isinstance(v, torch.Tensor) else v
                                     for k, v in components.items()},
                forward_seconds=forward_seconds, ranking_loss_seconds=loss_seconds,
                backward_seconds=backward_seconds, batch_preparation_seconds=preparation_seconds,
                encoder_batches_per_pass=len(packets), encoder_forward_passes=1,
                model_backward_calls=1, **storage.statistics(), **memory_metrics)
