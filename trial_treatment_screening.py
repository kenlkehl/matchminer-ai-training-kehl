"""Legacy training/audit adapter for the inference-owned combined cohort prompt."""

import hashlib
import json
from pathlib import Path

from remote_vllm_pool import OutputValidationError


from matchminer_ai.trials.cohort_schema import (
    COHORT_SCHEMA_VERSION, build_cohort_messages, load_cohort_prompt,
    parse_cohort_response, unwrap_json_fence,
)


def build_screening_messages(trial_text):
    """Legacy training entrypoint using the inference-owned combined prompt."""
    return build_cohort_messages(str(trial_text))


def parse_screening_output(final, finish_reason=None, *, allow_legacy=True):
    """Invalid/unfinished responses are errors, never negative decisions."""
    if finish_reason in {"length", "abort", "error"}:
        raise ValueError(f"Incomplete treatment-screen response: {finish_reason}")
    try:
        result = json.loads(unwrap_json_fence(final))
    except (TypeError, ValueError) as exc:
        raise OutputValidationError(
            f"Treatment screen must return a JSON object: {exc}"
        ) from exc
    if isinstance(result, dict) and "cancer_treatment_trial" in result:
        try:
            # Historical v1/v2 audits use strings. New requests require named
            # objects; read-only audit compatibility never relaxes live parsing.
            schema_version = COHORT_SCHEMA_VERSION
            if (allow_legacy and result.get("multiple_cohorts") == "yes"
                    and "common_eligibility_criteria" not in result):
                schema_version = "trial-treatment-cohorts-v1"
            elif (allow_legacy and result.get("multiple_cohorts") == "yes"
                  and isinstance(result.get("cohorts"), list)
                  and all(isinstance(c, str) for c in result["cohorts"])):
                schema_version = "trial-treatment-cohorts-v2"
            combined = parse_cohort_response(
                final, (finish_reason or "stop") if allow_legacy else finish_reason,
                schema_version=schema_version,
            )
        except ValueError as exc:
            raise OutputValidationError(str(exc)) from exc
        accepted = combined["cancer_treatment_trial"] == "yes"
        return {
            "is_cancer_treatment_trial": accepted,
            # This is a code-generated audit description, not an LLM rationale.
            "treatment_screen_rationale": (
                "Combined prompt classified this as a cancer treatment trial."
                if accepted else
                "Combined prompt did not establish direct cancer treatment intent."
            ),
            "trial_cohort_output": json.dumps(combined, ensure_ascii=False),
        }
    if not allow_legacy:
        raise OutputValidationError(
            "Live requests require the combined cancer_treatment_trial/cohort JSON schema"
        )
    # Read-only compatibility for historical Gemma screening audits.
    if not isinstance(result, dict) or type(result.get("is_cancer_treatment_trial")) is not bool:
        raise OutputValidationError("Treatment screen requires a JSON boolean is_cancer_treatment_trial")
    rationale = result.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise OutputValidationError("Treatment screen requires a nonempty rationale")
    return {"is_cancer_treatment_trial": result["is_cancer_treatment_trial"],
            "treatment_screen_rationale": rationale.strip()}


def screening_record(reasoning, final, finish_reason):
    return {**parse_screening_output(final, finish_reason, allow_legacy=False),
            "treatment_screen_reasoning": reasoning,
            "treatment_screen_output": final,
            "treatment_screen_finish_reason": finish_reason,
            "treatment_screen_status": "ok"}


def load_screening_results(trials, input_path, work_dir):
    """Reuse a complete audit against identical selected inputs, with independent extraction settings."""
    import pandas as pd
    from trial_cohort_format import read_clinical_csv

    source = Path(input_path)
    audit = read_clinical_csv(source)
    required = set(trials.columns) | {
        "is_cancer_treatment_trial", "treatment_screen_output",
        "treatment_screen_finish_reason", "treatment_screen_status", "treatment_screen_model",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Saved screening results lack required columns: {sorted(missing)}")
    if (audit.nct_id.isna().any() or not audit.nct_id.is_unique
            or set(audit.nct_id) != set(trials.nct_id)):
        raise ValueError("Saved screening results must cover every selected trial exactly once")
    audit = audit.set_index("nct_id", drop=False).loc[trials.nct_id].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            audit[trials.columns], trials.reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError("Saved screening inputs do not match the selected trial data and splits") from exc
    if not audit.treatment_screen_status.eq("ok").all():
        raise ValueError("Saved screening is incomplete; all trials must have status 'ok'")
    models = audit.treatment_screen_model.astype("string").str.strip()
    if models.isna().any() or models.eq("").any():
        raise ValueError("Saved screening results must retain the screening model for every trial")
    decisions = [parse_screening_output(row.treatment_screen_output, row.treatment_screen_finish_reason)
                 for row in audit.itertuples()]
    if not pd.api.types.is_bool_dtype(audit.is_cancer_treatment_trial.dtype):
        raise ValueError("Saved screening decisions must be boolean")
    if audit.is_cancer_treatment_trial.tolist() != [d["is_cancer_treatment_trial"] for d in decisions]:
        raise ValueError("Saved screening decisions disagree with the validated model responses")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = {"input_path": str(source.resolve()), "sha256": digest.hexdigest(),
                "rows": len(audit), "screening_models": sorted(models.unique().tolist())}
    path = Path(work_dir) / "screening_input_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Saved screening results changed; use a new --work-dir")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2))
        temporary.replace(path)
    selected = trials.loc[audit.is_cancer_treatment_trial.to_numpy()].copy()
    print(f"[screen] reused {len(audit)} validated decisions from {source}; "
          f"retained {len(selected)} trials; screening models: {identity['screening_models']}")
    return selected


def check_screening_manifest(args, reasoning_parser, work_dir):
    """Input identity is guarded by 0b's parent trial-selection manifest."""
    from vllm_reasoning_utils import resolve_chat_template_kwargs

    path = Path(work_dir) / "screening_manifest.json"
    identity = {
        "prompt_sha256": hashlib.sha256(load_cohort_prompt().encode()).hexdigest(),
        "model": args.model, "reasoning_parser": reasoning_parser,
        "enable_thinking": True,
        "max_model_len": args.max_model_len,
        "transport": "chat_completions" if reasoning_parser == "beam" else "completions",
        "sampling": {key: getattr(args, key) for key in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "repetition_penalty", "screening_max_tokens")},
    }
    if getattr(args, "tokenizer_model", None):
        identity["tokenizer_model"] = args.tokenizer_model
    if reasoning_parser == "beam":
        identity["chat_template_kwargs"] = resolve_chat_template_kwargs("beam")
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Treatment screening configuration changed; use a new --work-dir.")
    elif any(Path(work_dir).glob("*output_shard_*.parquet")):
        raise ValueError("Screening shards lack a manifest; use a new --work-dir.")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2))
        temporary.replace(path)
