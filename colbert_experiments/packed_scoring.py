"""Compatibility imports; production training and inference share one scorer."""
from matchminer_ai.embedding.packed_scoring import PackedTokens, score_pairs

__all__ = ["PackedTokens", "score_pairs"]
