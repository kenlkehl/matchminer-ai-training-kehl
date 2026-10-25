"""Shared native encoder plus the historical question-grouped training loss."""
import torch
from torch.nn import functional as F
from matchminer_ai.embedding.colbert import BASE_MODEL, ColBERT, maxsim

__all__ = ["BASE_MODEL", "ColBERT", "maxsim", "grouped_loss"]

def grouped_loss(scores, labels, temperature=.05):
    """All explicitly labeled positive/negative comparisons within one query/task.

Other queries and other questions never become implicit negatives.
"""
    labels = torch.as_tensor(labels, device=scores.device)
    if temperature <= 0 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Use positive temperature and binary labels')
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if not len(positives) or not len(negatives):
        raise ValueError('Each ranking group needs at least one known positive and negative')
    return F.softplus((negatives[None, :] - positives[:, None]) / temperature).mean()
