"""Run with python -m colbert_experiments; training also supports torchrun --module."""
import argparse
import asyncio
from pathlib import Path

from .data import read_jsonl, write_json
from .model import BASE_MODEL


def model_args(parser):
    parser.add_argument('--model', default=BASE_MODEL)
    parser.add_argument('--revision')
    parser.add_argument('--cache-dir')
    parser.add_argument('--query-length', type=int, default=8192)
    parser.add_argument('--document-length', type=int, default=8192)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Adapt existing full summaries and cohort CSV; preserve splits')
    prepare.add_argument('--patients', required=True)
    prepare.add_argument('--cohorts', required=True)
    prepare.add_argument('--output', required=True)
    sample = commands.add_parser('sample-v22', help='Sample v22 summaries and retain the full cohort corpus')
    sample.add_argument('--patients', required=True)
    sample.add_argument('--cohorts', required=True)
    sample.add_argument('--output', required=True)
    sample.add_argument('--train-count', type=int, default=256)
    sample.add_argument('--val-count', type=int, default=64)
    sample.add_argument('--seed', type=int, default=42)
    fixture = commands.add_parser('fixture', help='Create fabricated train/validation challenge cases')
    fixture.add_argument('--output', required=True)
    mine = commands.add_parser('mine', help='Bidirectional, per-question exact MaxSim + lexical/random union')
    model_args(mine)
    mine.add_argument('--objective', choices=['binary', 'gated_regression'], default='binary')
    mine.add_argument('--corpus', required=True)
    mine.add_argument('--output', required=True)
    mine.add_argument('--top-k', type=int, default=10)
    mine.add_argument('--lexical-k', type=int, default=5)
    mine.add_argument('--random-k', type=int, default=5)
    mine.add_argument('--candidate-limit', type=int)
    mine.add_argument('--proposal-strategy', choices=['random', 'lexical_random'], default='random')
    mine.add_argument('--cohort-query-limit', type=int, help='Limit cohort queries per split, retaining every cohort as a document')
    mine.add_argument('--extra-pairs', help='JSONL pairs from challenge/reference/baseline retrieval')
    mine.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    mine.add_argument('--seed', type=int, default=42)
    mine.add_argument('--device', default='cuda:0')
    mine.add_argument('--document-batch-size', type=int, default=16)
    label = commands.add_parser('label', help='Label each canonical pair on all five independent tasks')
    from remote_vllm_pool import add_remote_cli_args
    add_remote_cli_args(label)
    label.add_argument('--corpus', required=True)
    label.add_argument('--candidates', required=True)
    label.add_argument('--output', required=True)
    label.add_argument('--teacher-model', required=True, help='Served model ID')
    label.add_argument('--teacher-served-model', help='Endpoint model name when different from the recorded teacher identity; advertised root is verified')
    label.add_argument('--teacher-tokenizer', required=True, help='Underlying model for tokenizer and vendor sampling')
    label.add_argument('--cache-dir')
    label.add_argument('--max-tokens', type=int, default=8192)
    label.add_argument('--teacher-context', type=int, default=250000)
    label.add_argument('--pending-requests', type=int, default=16)
    label.add_argument('--gated-regression', action='store_true',
                       help='Ask potential eligibility first; only ask specificity after yes; save scores.jsonl')
    label.set_defaults(max_concurrent_per_server=4, initial_concurrent_per_server=2,
                       scheduler_adaptive_concurrency=True, request_timeout=7200, max_attempts=4)
    train = commands.add_parser('train', help='Train only explicit within-question positive/negative groups')
    model_args(train)
    train.add_argument('--corpus', required=True)
    train.add_argument('--labels', nargs='+', required=True)
    train.add_argument('--training-splits', nargs='+', choices=['train', 'val', 'test'], default=['train'],
                       help='Original source splits included in optimizer updates')
    train.add_argument('--require-complete-coverage', action=argparse.BooleanOptionalAction, default=False,
                       help='Require teacher labels covering every selected patient and cohort')
    train.add_argument('--output', required=True)
    train.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU is for explicit small smoke tests')
    train.add_argument('--objective', choices=['binary', 'gated_regression'], default='binary')
    train.add_argument('--documents-per-query', type=int, default=4)
    train.add_argument('--prediction-link', choices=['identity', 'sigmoid'], default='identity')
    train.add_argument('--steps', type=int, default=1000)
    train.add_argument('--accumulation', type=int, default=8)
    train.add_argument('--documents-per-class', type=int, default=2)
    train.add_argument('--learning-rate', type=float, default=1e-5)
    train.add_argument('--temperature', type=float, default=.05)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--save-steps', type=int, default=100)
    train.add_argument('--eval-every', type=int, default=0, help='Fixed train/validation ranking probes every N updates; zero disables')
    train.add_argument('--eval-groups-per-stratum', type=int, default=2)
    evaluate = commands.add_parser('evaluate', help='Evaluate against all judgments, including missed positives')
    evaluate.add_argument('--rankings', required=True)
    evaluate.add_argument('--labels', nargs='+', required=True)
    evaluate.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    evaluate.add_argument('--k', type=int, default=10)
    evaluate.add_argument('--output', required=True)
    probe = commands.add_parser('probe', help='Real full-context forward/backward and overflow check')
    model_args(probe)
    probe.add_argument('--tokens', type=int, default=8000)
    probe.add_argument('--device', default='cuda:0')
    probe.add_argument('--output', required=True)
    aggregate = commands.add_parser('aggregate-regression', help='Apply the gated scoring rule to verified existing single-question answers')
    aggregate.add_argument('--corpus', required=True)
    aggregate.add_argument('--candidates', required=True)
    aggregate.add_argument('--source-teacher', required=True)
    aggregate.add_argument('--output', required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        from .data import prepare
        prepare(args.patients, args.cohorts, args.output)
    elif args.command == 'sample-v22':
        from .data import sample_v22
        print(sample_v22(args.patients, args.cohorts, args.output,
                        train_count=args.train_count, val_count=args.val_count, seed=args.seed))
    elif args.command == 'fixture':
        from .fixtures import create_fixture
        create_fixture(args.output)
    elif args.command in ('mine', 'probe'):
        from .model import ColBERT
        model = ColBERT(args.model, revision=args.revision, cache_dir=args.cache_dir,
                       query_length=args.query_length, document_length=args.document_length,
                       checkpointing=args.command == 'probe')
        if args.command == 'mine':
            from .retrieval import mine
            mine(model, args.corpus, args.output, top_k=args.top_k, lexical_k=args.lexical_k,
                 random_k=args.random_k, candidate_limit=args.candidate_limit, extra_pairs=args.extra_pairs,
                 seed=args.seed, splits=args.splits, proposal_strategy=args.proposal_strategy,
                 cohort_query_limit=args.cohort_query_limit, document_batch_size=args.document_batch_size,
                 device=args.device, objective=args.objective)
        else:
            model.to(args.device)
            from .probe import probe
            write_json(args.output, probe(model, args.tokens))
    elif args.command == 'label':
        from .labeling import label
        if not args.scheduler_metrics_file:
            args.scheduler_metrics_file = str(Path(args.output) / 'concurrency_status.json')
        asyncio.run(label(args))
    elif args.command == 'train':
        from .training import train
        train(args)
    elif args.command == 'aggregate-regression':
        from .regression import import_verified_judgments
        print(import_verified_judgments(args.corpus, args.candidates, args.source_teacher, args.output))
    elif args.command == 'evaluate':
        from .retrieval import evaluate_rankings
        labels = [row for path in args.labels for row in read_jsonl(path) if row['split'] == args.split]
        write_json(args.output, evaluate_rankings(read_jsonl(args.rankings), labels, k=args.k))


if __name__ == '__main__':
    main()
