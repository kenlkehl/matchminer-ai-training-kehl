"""Run with python -m trialspace: prepare a TrialSpace corpus, or label its candidate pairs."""
import argparse
import asyncio
from pathlib import Path

from .label_concurrency import DEFAULT_CEILING, DEFAULT_INITIAL, DEFAULT_AUTO_WORKERS_MAX


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Adapt summaries and the cohort table; key and boilerplate texts')
    prepare.add_argument('--patients', required=True)
    prepare.add_argument('--cohorts', required=True)
    prepare.add_argument('--output', required=True)
    label = commands.add_parser('label', help='Label candidate pairs: the five match labels, or boilerplate')
    from remote_vllm_pool import add_remote_cli_args
    add_remote_cli_args(label)
    label.add_argument('--corpus', required=True)
    label.add_argument('--candidates', required=True)
    label.add_argument('--output', required=True)
    label.add_argument('--task', choices=['match', 'boilerplate', 'criteria'], default='match',
                       help='match: the five match labels in one request per pair, scored 0-6 into scores.jsonl; '
                            'boilerplate: whether the patient clearly fails the cohort\'s boilerplate criteria; '
                            'criteria: one request per criteria line of sampled pairs (yes = does not clearly fail it)')
    label.add_argument('--teacher-model', required=True, help='Served model ID')
    label.add_argument('--teacher-served-model', help='Endpoint model name when different from the recorded teacher identity; advertised root is verified')
    label.add_argument('--teacher-tokenizer', required=True, help='Underlying model for tokenizer and vendor sampling')
    label.add_argument('--cache-dir')
    label.add_argument('--max-tokens', type=int, default=32768)
    label.add_argument('--teacher-context', type=int, default=250000)
    label.add_argument('--pending-requests', type=int, default=0,
                       help='Pairs in flight; 0 automatically follows live endpoint capacity')
    label.add_argument('--max-auto-pending-requests', type=int, default=DEFAULT_AUTO_WORKERS_MAX,
                       help='Upper bound on automatic pair workers; explicit --pending-requests overrides it')
    label.set_defaults(max_concurrent_per_server=DEFAULT_CEILING, initial_concurrent_per_server=DEFAULT_INITIAL,
                       scheduler_adaptive_concurrency=True, request_timeout=7200, max_attempts=4)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        from .data import prepare
        prepare(args.patients, args.cohorts, args.output)
    elif args.command == 'label':
        from .labeling import label
        if not args.scheduler_metrics_file:
            args.scheduler_metrics_file = str(Path(args.output) / 'concurrency_status.json')
        asyncio.run(label(args))


if __name__ == '__main__':
    main()
