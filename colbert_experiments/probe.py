"""Exercise complete long queries/documents, backward, and explicit overflow."""
import time
import torch

from .model import grouped_loss


def probe(model, tokens=8000):
    if tokens < 16 or tokens + 3 > min(model.query_length, model.document_length):
        raise ValueError('Probe content plus special tokens must fit both configured lengths')
    # Select an ordinary one-token word; preserve a meaningful end-of-input fact.
    body = ' history' * (tokens - 20)
    query = 'Patient' + body + ' Documented EGFR L858R mutation. No brain metastases.'
    positive = 'Eligibility' + body + ' Required EGFR L858R mutation. Exclude brain metastases.'
    negative = 'Eligibility' + body + ' Required absence of EGFR mutations. Require brain metastases.'
    lengths = [model.tokens(t, is_query=i == 0)['input_ids'].shape[1]
               for i, t in enumerate((query, positive, negative))]
    start = time.monotonic()
    model.train()
    torch.cuda.reset_peak_memory_stats(model.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        scores = model(query, [positive, negative])
        loss = grouped_loss(scores, [1, 0])
    loss.backward()
    grad = model.projection.weight.grad
    if not torch.isfinite(grad).all() or not grad.abs().sum():
        raise AssertionError('Missing or nonfinite projection gradients')
    overflow = False
    try:
        model.tokens(' history' * (model.query_length + 10), is_query=True)
    except ValueError as error:
        overflow = 'no truncation' in str(error)
    if not overflow:
        raise AssertionError('Overflow was not explicitly rejected')
    return dict(input_lengths=lengths, score=scores.detach().cpu().tolist(), loss=float(loss.detach()),
        projection_gradient_norm=float(grad.norm()), overflow_rejected=overflow,
        elapsed_seconds=time.monotonic() - start, gpu=torch.cuda.get_device_name(model.device),
        max_memory_allocated=torch.cuda.max_memory_allocated(model.device), model=model.fingerprint())
