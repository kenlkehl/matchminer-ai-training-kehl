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


def available_host_bytes():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class ActivationStorage:
    """Retain saved tensors on GPU or spill to pinned RAM without recomputation.

    Auto's soft limit leaves half of initially free device memory for forward
    temporaries, restored tensors and backward. It is not a hard VRAM limit.
    Parameter/buffer storage remains on device because the model owns it anyway.
    """
    def __init__(self, model, device, *, mode='auto', gpu_budget_gb=0):
        if mode not in ('auto', 'gpu', 'cpu') or not math.isfinite(gpu_budget_gb) or gpu_budget_gb < 0:
            raise ValueError('Expected auto/gpu/cpu activation storage and nonnegative GPU budget')
        self.mode, self.device = mode, torch.device(device)
        self.offloaded_bytes = self.offloaded_tensors = 0
        self.next_host_check = 0
        self.host_copies = {}
        self.limit = None
        self.persistent = {t.untyped_storage().data_ptr() for t in
                           [*model.parameters(), *model.buffers()] if t.device.type == 'cuda'}
        if mode == 'auto' and self.device.type == 'cuda':
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
        if (tensor.device.type != 'cuda' or self.mode == 'gpu'
                or tensor.untyped_storage().data_ptr() in self.persistent
                or (self.mode == 'auto' and torch.cuda.memory_allocated(self.device) < self.limit)):
            return tensor.detach()
        size = tensor.numel() * tensor.element_size()
        if self.offloaded_bytes + size >= self.next_host_check:
            available = available_host_bytes()
            if available is not None and available < size + 8 * 2**30:
                raise MemoryError('Retained activations would leave less than 8 GiB available host RAM. '
                                  'Reduce the effective batch or raise the GPU retention budget; '
                                  'the trainer will not silently replay or discard examples.')
            self.next_host_check = self.offloaded_bytes + size + 2**30
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
        host.copy_(tensor.detach(), non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(tensor.device))
        self.offloaded_bytes += host.numel() * host.element_size()
        self.offloaded_tensors += 1
        saved = tensor.device, host, ready
        # Repeated uses of an unchanged tensor (notably attention masks) share
        # one host copy. Weak identity plus version prevents pointer-reuse and
        # in-place-update collisions without keeping the source tensor on GPU.
        if key is not None:
            self.host_copies[key] = weakref.ref(tensor), saved
        return saved

    @staticmethod
    def unpack(saved):
        if isinstance(saved, torch.Tensor):
            return saved
        device, host, ready = saved
        torch.cuda.current_stream(device).wait_event(ready)
        return host.to(device, non_blocking=True)

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
                              activation_storage='auto', activation_gpu_budget_gb=0):
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
    if not torch.isfinite(loss_scores).all():
        raise ValueError('Nonfinite retained MaxSim score')
    started = time.monotonic()
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
    torch.autograd.backward(predictions, gradients)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    backward_seconds = time.monotonic() - started
    if should_pause():
        return None
    return dict(loss=loss.detach(), **{k: v.detach() if isinstance(v, torch.Tensor) else v
                                     for k, v in components.items()},
                forward_seconds=forward_seconds, ranking_loss_seconds=loss_seconds,
                backward_seconds=backward_seconds, batch_preparation_seconds=preparation_seconds,
                encoder_batches_per_pass=len(packets), encoder_forward_passes=1,
                model_backward_calls=1, **storage.statistics())
