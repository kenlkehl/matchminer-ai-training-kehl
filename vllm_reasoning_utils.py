"""Shared helpers for picking and running a vLLM reasoning parser.

One place that maps model names to vLLM reasoning-parser names, adds the
standard CLI flag, and parses a raw generated string via the appropriate
parser. Every vLLM-touching script in this repo goes through this module so
swapping models (e.g. Gemma-4 -> Qwen3.6-27B) is a one-flag change.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

MODEL_TO_PARSER: dict[str, str] = {
    "google/gemma-4-E2B":     "gemma4",
    "google/gemma-4-E2B-it":  "gemma4",
    "google/gemma-4-31b-it":  "gemma4",
    "Qwen/Qwen3.6-27B":       "qwen3",
    "Qwen/Qwen3.6-27B-FP8":   "qwen3",
    "Qwen/Qwen3.5-9B":        "qwen3",
    "Qwen/Qwen3.5-35B-A3B":   "qwen3",
    "openai/gpt-oss-20b":     "openai_gptoss",
    "openai/gpt-oss-120b":    "openai_gptoss",
}

_SUBSTRING_FALLBACKS: list[tuple[str, str]] = [
    ("gemma",    "gemma4"),
    ("qwen",     "qwen3"),
    ("gpt-oss",  "openai_gptoss"),
    ("gptoss",   "openai_gptoss"),
    ("deepseek", "deepseek_r1"),
]


def resolve_parser_name(model: str, explicit: Optional[str] = None) -> str:
    """Pick a vLLM reasoning-parser name for a given model.

    If ``explicit`` is set and not ``"auto"``, it wins. Otherwise look up the
    model in MODEL_TO_PARSER, then fall back to case-insensitive substring
    match on known families.
    """
    if explicit and explicit != "auto":
        return explicit
    if model in MODEL_TO_PARSER:
        return MODEL_TO_PARSER[model]
    lowered = model.lower()
    for needle, parser in _SUBSTRING_FALLBACKS:
        if needle in lowered:
            return parser
    raise ValueError(
        f"Cannot infer reasoning parser for model {model!r}. "
        f"Pass --reasoning-parser explicitly. Known models: "
        f"{sorted(MODEL_TO_PARSER.keys())}"
    )


def parse_reasoning_output(text: str, parser_name: str, tokenizer) -> Tuple[str, str]:
    """Split raw model output into (reasoning, answer) using vLLM's parser.

    ``gemma4`` uses :func:`vllm.reasoning.gemma4_utils.parse_thinking_output`
    which also strips trailing ``<turn|>`` / ``<eos>`` sentinels — behavior
    the class-based ``extract_reasoning`` skips. ``openai_gptoss`` goes
    through vLLM's harmony helper (its ``extract_reasoning`` raises
    NotImplementedError). Everything else goes through the standard
    :class:`ReasoningParser` class API.
    """
    if parser_name == "gemma4":
        from vllm.reasoning.gemma4_utils import parse_thinking_output
        parsed = parse_thinking_output(text)
        reasoning, content = parsed["thinking"], parsed["answer"]
    elif parser_name == "openai_gptoss":
        from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output
        reasoning, content = parse_chat_output(text)
    else:
        from vllm.reasoning import (
            ReasoningParserManager,
            register_lazy_reasoning_parsers,
        )
        register_lazy_reasoning_parsers()
        cls = ReasoningParserManager.get_reasoning_parser(parser_name)
        parser = cls(tokenizer)
        reasoning, content = parser.extract_reasoning(text, request=None)
    return (reasoning or "").strip(), (content or "").strip()


def add_reasoning_cli_args(ap: argparse.ArgumentParser) -> None:
    """Add the standard ``--reasoning-parser`` flag.

    Default ``"auto"`` routes through :func:`resolve_parser_name` using the
    script's ``--model`` value.
    """
    ap.add_argument(
        "--reasoning-parser", "--reasoning_parser",
        dest="reasoning_parser",
        default="auto",
        help=(
            "vLLM reasoning parser name (default: auto = infer from --model). "
            "Examples: gemma4, qwen3, openai_gptoss, deepseek_r1."
        ),
    )
