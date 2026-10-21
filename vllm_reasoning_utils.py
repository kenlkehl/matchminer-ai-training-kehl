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
    ("nemotron-3.5-super-vl", "nemotron_v3"),
    ("gemma",         "gemma4"),
    ("qwen",          "qwen3"),
    ("gpt-oss",       "openai_gptoss"),
    ("gptoss",        "openai_gptoss"),
    # OncoReasoning is built on LFM 2.5, which emits DeepSeek-R1-style
    # <think>...</think> reasoning tags — route both to the deepseek_r1 parser.
    ("oncoreasoning", "deepseek_r1"),
    ("deepseek",      "deepseek_r1"),
]


def resolve_parser_name(model: str, explicit: Optional[str] = None) -> str:
    """Pick a vLLM reasoning-parser name for a given model.

    If ``explicit`` is set and not ``"auto"``, it wins. Otherwise consult the
    shared model profiles, then MODEL_TO_PARSER and case-insensitive substring
    fallbacks for other known families.
    """
    if explicit and explicit != "auto":
        return explicit
    from matchminer_ai.llm.model_profiles import resolve_model_profile

    profile = resolve_model_profile(model)
    if profile is not None and profile.reasoning_parser is not None:
        return profile.reasoning_parser
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


def resolve_tool_call_parser(model: str, explicit: Optional[str] = None) -> Optional[str]:
    """Use the inference package's publisher-backed tool-parser registry."""
    from matchminer_ai.llm.reasoning import resolve_tool_call_parser as resolve
    return resolve(model, explicit)


SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
)


def resolve_sampling_defaults(model: str, *, fallback=None, enable_thinking=True, **overrides) -> dict:
    """Text sampling from the shared inference profile, with explicit overrides.

    Use None for unspecified CLI values; callers can supply legacy defaults
    separately for unregistered models. No task output budget is inferred.
    """
    from matchminer_ai.llm.sampling import vendor_defaults
    defaults, _ = vendor_defaults(model, template={"enable_thinking": enable_thinking})
    # A recognized family must not inherit another task/model's penalties.
    # These are neutral controls, not additional publisher recommendations.
    neutral = dict(min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0) if defaults else {}
    return {**(fallback or {}), **neutral, **defaults,
            **{key: value for key, value in overrides.items() if value is not None}}


def add_sampling_cli_args(parser) -> None:
    """Unspecified CLI values select model defaults; explicit values win."""
    for key in SAMPLING_KEYS:
        flags = list(dict.fromkeys(("--" + key.replace("_", "-"), "--" + key)))
        parser.add_argument(*flags, dest=key, type=int if key == "top_k" else float,
                            default=None, help="Default: detect from model family.")


def sampling_from_args(args, *, model=None, fallback=None) -> dict:
    """Resolve CLI overrides consistently for local and remote generation."""
    model = model or getattr(args, "tokenizer_model", None) or getattr(args, "model", None) or getattr(args, "model_name", None)
    return resolve_sampling_defaults(
        model, fallback=fallback,
        **{key: getattr(args, key, None) for key in SAMPLING_KEYS},
    )


def parse_reasoning_output(
    text: str, parser_name: str, tokenizer, *, enable_thinking: Optional[bool] = None,
) -> Tuple[str, str]:
    """Split raw model output into (reasoning, answer) using vLLM's parser.

    ``gemma4`` uses :func:`vllm.reasoning.gemma4_utils.parse_thinking_output`
    which also strips trailing ``<turn|>`` / ``<eos>`` sentinels — behavior
    the class-based ``extract_reasoning`` skips. ``openai_gptoss`` goes
    through vLLM's harmony helper (its ``extract_reasoning`` raises
    NotImplementedError). Everything else goes through the standard
    :class:`ReasoningParser` class API.
    """
    from matchminer_ai.llm.reasoning import parse_reasoning_output as parse

    return parse(text, parser_name=parser_name, tokenizer=tokenizer,
                 enable_thinking=enable_thinking)


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
            "Examples: gemma4, qwen3, openai_gptoss, deepseek_r1, beam."
        ),
    )
