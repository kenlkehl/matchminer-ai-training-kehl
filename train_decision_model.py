#!/usr/bin/env python3
"""Fine-tune the decision model (LiquidAI d1-2B) on the teacher's labels of its seven question families.

``prepare`` turns every round's teacher labels into one row per *state* (``examples.jsonl``): a
family, the patient and cohort ids, and the yes/no answer of each of the state's questions. The
states and questions themselves are rendered at training time from the corpus with the inference
package's builders (``matchminer_ai.matching.decision_model``), so training and inference read the
same text:

- ``match`` (questions 1-5): the five match labels of a pair's ``scores.jsonl`` row, about the
  patient's key section and the cohort's key criteria. ``reasonableness`` yes = potentially eligible.
- ``boilerplate`` (question 6): one question about the boilerplate sections; the boilerplate teacher's
  ``excluded`` inverted, so yes = potentially eligible (not clearly failing any boilerplate criterion).
- ``criterion`` (question 7): one state per patient and set of criteria lines, the complete summary
  and the numbered criteria, with one question per labeled line. **Yes = the patient does not clearly
  fail the line; no = an inclusion line clearly not met or an exclusion line clearly met.**

Later rounds win a conflict. States longer than --max-length with one question are dropped and
counted. ``--preview N`` also writes N fully rendered prompts with their answers (``preview.jsonl``).

``train`` fully fine-tunes the model (no LoRA) with transformers' Trainer. Each state's questions are
shuffled (seeded per epoch) and packed after the state exactly as the model serves them (its own
``SlotReader._sequences``: at most its trained set size, 16, per sequence, and at most --max-length
tokens). d1-2B reads a state with ``set`` attention (both ways, with its questions, on its
full-attention layers), so the answer is never appended to the input: the loss is the cross-entropy
of the ``yes``/``no`` token at each question's answer slot and nothing else. The vision tower and
projector are frozen. Writes the model, its own code, tokenizer, LICENSE and a modification notice,
``calibration.json`` (temperature 1.0: the base temperature was fitted to the base model) and
``decision_contract.json`` to --output-model. Run under torchrun for several GPUs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from trialspace.data import digest, full_summary, load_corpus, read_jsonl, write_json, write_jsonl

BASE_MODEL = 'LiquidAI/d1-2B'
BASE_REVISION = '652f920e544ea37f81e4b0ce89b81d4ed0da90b1'
FAMILIES = ('match', 'boilerplate', 'criterion')


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #


def render(row, patients, cohorts):
    """(state, {name: instructions}, {name: answer}) of one example row."""
    from matchminer_ai.matching import decision_model as d
    p, c = patients[row['patient_id']], cohorts[row['cohort_id']]
    answers = row['answers']
    if row['family'] == 'match':
        questions = d.match_questions()
        return d.match_state(p['text'], c['text']), questions, {name: answers[name] for name in questions}
    if row['family'] == 'boilerplate':
        return d.boilerplate_state(p.get('boilerplate') or '', c['boilerplate']), {d.BOILERPLATE: d.boilerplate_question()}, \
            {d.BOILERPLATE: answers[d.BOILERPLATE]}
    items = c['items']
    questions = {name: d.criterion_question(items[int(name.rsplit('_', 1)[1]) - 1]) for name in answers}
    return d.criteria_state(full_summary(p), items), questions, dict(answers)


def section_tokens(tokenizer, texts):
    counts = {}
    unique = list(dict.fromkeys(texts))
    for start in range(0, len(unique), 2048):
        chunk = unique[start:start + 2048]
        for text, ids in zip(chunk, tokenizer(chunk, add_special_tokens=False)['input_ids']):
            counts[text] = len(ids)
    return counts


def prepare(args):
    from transformers import AutoTokenizer
    from matchminer_ai.matching import decision_model as d
    from trialspace_training import criteria_digest

    patients, cohorts = load_corpus(args.corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    examples, sources = {}, dict(match=[], boilerplate=[], criterion=[])
    conflicts = Counter()

    def add(key, row):
        if key in examples and examples[key]['answers'] != row['answers']:
            conflicts[row['family']] += 1
        examples[key] = row  # later rounds win

    for directory in args.match_labels:
        sources['match'].append(str(Path(directory).resolve()))
        for row in read_jsonl(Path(directory) / 'scores.jsonl'):
            add(('match', row['patient_id'], row['cohort_id']), dict(
                family='match', patient_id=row['patient_id'], cohort_id=row['cohort_id'], answers=dict(row['answers'])))
    for directory in args.boilerplate_labels:
        sources['boilerplate'].append(str(Path(directory).resolve()))
        for row in read_jsonl(Path(directory) / 'labels.jsonl'):
            if row.get('task') != 'boilerplate' or type(row.get('excluded')) is not bool:
                raise ValueError('Expected boilerplate labels with a boolean excluded answer')
            # yes = potentially eligible: the teacher did not find the patient clearly excluded
            add(('boilerplate', row['patient_id'], row['cohort_id']), dict(
                family='boilerplate', patient_id=row['patient_id'], cohort_id=row['cohort_id'],
                answers={d.BOILERPLATE: 'no' if row['excluded'] else 'yes'}))
    lines = {}
    for directory in args.criteria_labels:
        sources['criterion'].append(str(Path(directory).resolve()))
        for row in read_jsonl(Path(directory) / 'labels.jsonl'):
            if row.get('task') != 'criteria':
                raise ValueError('Expected per-criterion labels')
            if criteria_digest(cmap[row['cohort_id']]['items']) != row['criteria_digest']:
                raise ValueError('A per-criterion label was made for different criteria than the corpus holds')
            state = lines.setdefault((row['patient_id'], row['criteria_digest']), dict(
                family='criterion', patient_id=row['patient_id'], cohort_id=row['cohort_id'], answers={}))
            name = f"{d.CRITERION}_{int(row['number'])}"
            if state['answers'].get(name, row['criterion']['answer']) != row['criterion']['answer']:
                conflicts['criterion'] += 1
            state['answers'][name] = row['criterion']['answer']  # yes = does not clearly fail the line
    for (patient_id, criteria), row in lines.items():
        row['answers'] = dict(sorted(row['answers'].items(), key=lambda kv: int(kv[0].rsplit('_', 1)[1])))
        examples[('criterion', patient_id, criteria)] = row

    # Token lengths: match and boilerplate states from each section's count (tokenized once per entity,
    # exact up to the few tokens a joint encoding merges); criterion states and questions exactly.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=False)
    rows = [examples[k] for k in sorted(examples, key=lambda k: (FAMILIES.index(k[0]), k[1], str(k[2])))]

    def n(text):
        return len(tokenizer(text, add_special_tokens=False)['input_ids'])

    sections = section_tokens(tokenizer, [x.strip() for x in [p['text'] for p in patients] + [c['text'] for c in cohorts]
                                          + [p.get('boilerplate') or '' for p in patients] + [c['boilerplate'] for c in cohorts]])
    overhead = dict(match=n(d.match_state('', '')), boilerplate=n(d.boilerplate_state('', '')))
    fixed = dict(match=[n(q) for q in d.match_questions().values()], boilerplate=[n(d.boilerplate_question())])
    budget = args.max_length - 64  # markup: turn headers, "QUESTION:", the reply instruction, the answer slot
    kept, dropped = [], Counter()
    for row in rows:
        p, c = pmap[row['patient_id']], cmap[row['cohort_id']]
        if row['family'] == 'match':
            state_tokens, question_tokens = overhead['match'] + sections[p['text'].strip()] + sections[c['text'].strip()], \
                fixed['match']
        elif row['family'] == 'boilerplate':
            state_tokens = overhead['boilerplate'] + sections[(p.get('boilerplate') or '').strip()] + \
                sections[c['boilerplate'].strip()]
            question_tokens = fixed['boilerplate']
        else:
            state, questions, _ = render(row, pmap, cmap)
            state_tokens = n(state)
            question_tokens = [len(ids) for ids in tokenizer(list(questions.values()), add_special_tokens=False)['input_ids']]
        question_tokens = [q + 16 for q in question_tokens]
        if state_tokens + max(question_tokens) > budget:
            dropped[row['family']] += 1
            continue
        kept.append(dict(row, tokens=state_tokens + sum(question_tokens)))
    output = Path(args.output_dir)
    write_jsonl(output / 'examples.jsonl', kept)
    if args.preview:
        rng = random.Random(args.seed)
        preview = []
        for row in rng.sample(kept, min(args.preview, len(kept))):
            state, questions, answers = render(row, pmap, cmap)
            preview.append(dict(family=row['family'], patient_id=row['patient_id'], cohort_id=row['cohort_id'],
                                state=state, questions=[dict(name=n, instructions=q, answer=answers[n])
                                                        for n, q in questions.items()]))
        write_jsonl(output / 'preview.jsonl', preview)
    answers = {family: Counter() for family in FAMILIES}
    for row in kept:
        for name, value in row['answers'].items():
            answers[row['family']][(name.split('_')[0] if row['family'] == 'criterion' else name, value)] += 1
    manifest = dict(
        corpus=str(Path(args.corpus).resolve()), label_sources=sources, base_model=args.base_model,
        base_revision=args.base_revision, questions_sha256=d.questions_digest(), max_length=args.max_length,
        states={family: sum(r['family'] == family for r in kept) for family in FAMILIES},
        questions={family: sum(len(r['answers']) for r in kept if r['family'] == family) for family in FAMILIES},
        answers={family: {f'{n}:{v}': c for (n, v), c in sorted(counts.items())} for family, counts in answers.items()},
        dropped_over_max_length=dict(dropped), conflicting_rounds=dict(conflicts),
        tokens=dict(total=sum(r['tokens'] for r in kept), max=max((r['tokens'] for r in kept), default=0)),
        polarity=('every question: yes = potentially eligible / not clearly failing; match questions 2-5: yes = the '
                  'cohort specifically requires that dimension and the patient documentedly satisfies it'))
    write_json(output / 'prepared.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('states', 'questions', 'dropped_over_max_length', 'tokens')}), flush=True)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def d1_module(model, name):
    import importlib
    return importlib.import_module(type(model).__module__.rsplit('.', 1)[0] + '.' + name)


def packer_for(model, tokenizer, attention):
    """The checkpoint's own request packing (``SlotReader._sequences``) without its inference engine."""
    runner, prompt = d1_module(model, 'runner'), d1_module(model, 'prompt')
    packer = runner.SlotReader.__new__(runner.SlotReader)
    packer.tokenizer = tokenizer
    packer._serve(model.config, attention=attention)
    bos = getattr(tokenizer, 'bos_token', None)
    packer.bos = bos if isinstance(bos, str) else ''
    packer.lead = prompt.default_lead(model.config.model_type, tokenizer)
    packer.state_style, packer.system, packer.option_style = prompt.DEFAULT_STATE_STYLE, prompt.DEFAULT_SYSTEM, 'desc'
    packer.processor = None
    return packer


class Collator:
    """Renders, shuffles and packs a batch of states into right-padded sequences and their answer slots."""

    def __init__(self, packer, prompt, corpus, *, max_length, seed):
        self.packer, self.prompt, self.max_length, self.seed = packer, prompt, max_length, seed
        patients, cohorts = load_corpus(corpus)
        self.patients, self.cohorts = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
        yes, no = prompt.readout_ids(packer.tokenizer, prompt.Noul(''))
        self.answer_ids = {'yes': yes[0], 'no': no[0]}  # the canonical lowercase tokens
        self.pad = packer.tokenizer.pad_token_id or 0

    def sequences(self, state, names, questions, set_size):
        qs = [self.prompt.as_question(dict(type='noul', instructions=questions[n])) for n in names]
        packed = self.packer._sequences([(state, qs, [])], set_size=set_size)
        if set_size > 1 and any(len(s[2]) > self.max_length for s in packed):
            return self.sequences(state, names, questions, max(1, set_size // 2))
        return packed

    def __call__(self, rows):
        import torch
        ids, state_lens, groups, slot_rows, slot_positions, targets = [], [], [], [], [], []
        for row in rows:
            example = json.loads(row['example'])
            state, questions, answers = render(example, self.patients, self.cohorts)
            names = list(questions)
            random.Random(f"{self.seed}-{row['epoch']}-{row['index']}").shuffle(names)  # a new order every epoch
            for _, first, seq, state_len, question, ends, _ in self.sequences(state, names, questions,
                                                                              self.packer.set_size):
                r = len(ids)
                ids.append(seq)
                state_lens.append(state_len)
                groups.append(question)
                for k, slots in enumerate(ends):
                    slot_rows.append(r)
                    slot_positions.append(slots[-1])  # a noul question has one slot, its last token
                    targets.append(self.answer_ids[answers[names[first + k]]])
        width = max(map(len, ids))
        batch = dict(input_ids=torch.tensor([s + [self.pad] * (width - len(s)) for s in ids]),
                     attention_mask=torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in ids]),
                     slot_rows=torch.tensor(slot_rows), slot_positions=torch.tensor(slot_positions),
                     targets=torch.tensor(targets))
        if state_lens[0] is not None:
            batch['state_len'] = torch.tensor(state_lens)
        if groups[0] is not None:
            batch['question'] = torch.tensor([g + [-1] * (width - len(g)) for g in groups])
        return batch


def train(args):
    import types

    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    from matchminer_ai.matching.decision_model import questions_digest

    data = Path(args.data_dir)
    prepared = json.loads((data / 'prepared.json').read_text())
    if prepared['questions_sha256'] != questions_digest():
        raise ValueError('The decision-model questions changed since prepare; prepare again')
    rows = read_jsonl(data / 'examples.jsonl')
    if args.max_examples:
        rows = random.Random(args.seed).sample(rows, min(args.max_examples, len(rows)))
    if not rows:
        raise ValueError('No examples to train on')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=True,
                                      dtype=torch.float32)
    if args.attention:
        model.config.d1_attention = args.attention
    attention = model.config.d1_attention
    for name in ('vision_tower', 'multi_modal_projector'):  # text only: the picture encoder stays as trained
        part = getattr(getattr(model, 'model', model), name, None)
        if part is not None:
            part.requires_grad_(False)
    model.config.use_cache = False
    slots = d1_module(model, 'slots')

    def slot_forward(self, input_ids, attention_mask, slot_rows, slot_positions, state_len=None, question=None, **_):
        """Logits over the vocabulary at the answer slots only (the training forward; DDP wraps it)."""
        return slots.slot_logits(self, input_ids, attention_mask, slot_positions, state_len, question, slot_rows).codes

    model.forward = types.MethodType(slot_forward, model)
    collator = Collator(packer_for(model, tokenizer, attention), d1_module(model, 'prompt'), prepared['corpus'],
                        max_length=args.max_length, seed=args.seed)
    epochs = max(1, int(round(args.epochs)))
    dataset = Dataset.from_dict(dict(
        example=[json.dumps(r) for _ in range(epochs) for r in rows],
        length=[r['tokens'] for _ in range(epochs) for r in rows],
        index=[i for _ in range(epochs) for i in range(len(rows))],
        epoch=[e for e in range(epochs) for _ in rows]))

    class SlotTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop('targets')
            logits = model(**inputs)
            loss = F.cross_entropy(logits.float(), targets)  # the answer slots only
            return (loss, logits) if return_outputs else loss

    output = Path(args.output_model).expanduser().resolve()
    work = Path(args.work_dir or output.with_name(output.name + '_training')).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    trainer = SlotTrainer(model=model, train_dataset=dataset, data_collator=collator, args=TrainingArguments(
        output_dir=str(work), learning_rate=args.learning_rate, num_train_epochs=1,  # epochs are copies, reshuffled
        max_steps=args.max_steps or -1, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_steps=args.warmup,
        lr_scheduler_type='cosine', weight_decay=0.01, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs=dict(use_reentrant=False), train_sampling_strategy='group_by_length', length_column_name='length',
        remove_unused_columns=False, logging_steps=args.logging_steps, save_steps=args.save_steps,
        save_total_limit=2, report_to='none', seed=args.seed, dataloader_num_workers=args.loader_workers,
        ddp_find_unused_parameters=False))
    trainer.model_accepts_loss_kwargs = False  # compute_loss returns the batch mean
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(work)))
    if trainer.is_world_process_zero():
        save(trainer, model, tokenizer, output, args, prepared, len(rows), attention)


def save(trainer, model, tokenizer, output, args, prepared, examples, attention):
    from huggingface_hub import snapshot_download

    import torch

    model.to(torch.bfloat16)  # trained with fp32 master weights; saved in the base model's dtype
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    base = Path(snapshot_download(args.base_model, revision=args.base_revision)) if not Path(args.base_model).is_dir() \
        else Path(args.base_model)
    # save_pretrained rewrites auto_map to the config class's own module, which skips the base's
    # native_lfm3_vl (it registers the text config): keep the base checkpoint's entry points.
    config = json.loads((output / 'config.json').read_text())
    config['auto_map'] = json.loads((base / 'config.json').read_text())['auto_map']
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for path in base.glob('*.py'):  # the System One code the checkpoint loads with (trust_remote_code)
        shutil.copy(path, output / path.name)
    for name in ('processor_config.json', 'LICENSE'):
        if (base / name).exists():
            shutil.copy(base / name, output / name)
    calibration = json.loads((base / 'calibration.json').read_text())
    calibration.update(default=1.0, temperatures={})  # the base temperature was fitted to the base model
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=1) + '\n')
    contract = dict(base_model=args.base_model, base_revision=args.base_revision, attention=attention,
                    set_size=getattr(model.config, 'd1_set_size', None), max_length=args.max_length,
                    questions_sha256=prepared['questions_sha256'], corpus=prepared['corpus'],
                    label_sources=prepared['label_sources'], states=prepared['states'], questions=prepared['questions'],
                    trained_examples=examples, epochs=args.epochs, learning_rate=args.learning_rate,
                    batch_size_per_device=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
                    seed=args.seed, loss='cross-entropy of the yes/no token at each answer slot only',
                    holdout='none (all synthetic pairs train)', polarity=prepared['polarity'])
    contract['fingerprint'] = digest(contract)
    (output / 'decision_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    (output / 'NOTICE.md').write_text(
        f"# Modification notice\n\nThis model was fine-tuned by MatchMiner-AI (`train_decision_model.py`) from "
        f"[{args.base_model}](https://huggingface.co/{args.base_model}) at revision {args.base_revision}. Its weights "
        "and `calibration.json` (temperature 1.0) differ from the original; the System One code files are copied "
        "unchanged. The LFM Open License v1.0 (`LICENSE`) applies, including its commercial-use limitation.\n\n"
        "Research use only: its answers are screening signals for human review, not eligibility decisions.\n")
    print(f'saved the decision model ({examples} states) to {output}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare', help='Teacher labels -> examples.jsonl (one row per state)')
    a.add_argument('--corpus', required=True, help='trialspace_training.py prepare output')
    a.add_argument('--match-labels', nargs='*', default=[], help='Rounds\' match label directories (scores.jsonl)')
    a.add_argument('--boilerplate-labels', nargs='*', default=[], help='Rounds\' boilerplate label directories')
    a.add_argument('--criteria-labels', nargs='*', default=[], help='Rounds\' per-criterion label directories')
    a.add_argument('--output-dir', required=True)
    a.add_argument('--base-model', default=BASE_MODEL)
    a.add_argument('--base-revision', default=BASE_REVISION)
    a.add_argument('--max-length', type=int, default=10000)
    a.add_argument('--preview', type=int, default=20, help='Fully rendered examples written to preview.jsonl')
    a.add_argument('--seed', type=int, default=42)
    b = sub.add_parser('train', help='Fine-tune the decision model on prepared examples')
    b.add_argument('--data-dir', required=True, help='prepare --output-dir')
    b.add_argument('--output-model', required=True)
    b.add_argument('--work-dir', help='Checkpoints; default <output-model>_training beside it')
    b.add_argument('--base-model', default=BASE_MODEL)
    b.add_argument('--base-revision', default=BASE_REVISION)
    b.add_argument('--attention', choices=['set', 'causal'], help="Default: the checkpoint's own (d1-2B: set)")
    b.add_argument('--max-length', type=int, default=10000)
    b.add_argument('--epochs', type=float, default=1)
    b.add_argument('--max-steps', type=int, help='Optimizer steps (smoke tests); default: the epochs')
    b.add_argument('--max-examples', type=int, help='Train on a seeded sample of this many states (smoke tests)')
    b.add_argument('--batch-size', type=int, default=8, help='States per GPU per micro-batch')
    b.add_argument('--gradient-accumulation-steps', type=int, default=4)
    b.add_argument('--learning-rate', type=float, default=1e-5)
    b.add_argument('--warmup', type=float, default=0.03, help='A float below 1 is a fraction of the steps')
    b.add_argument('--logging-steps', type=int, default=25)
    b.add_argument('--save-steps', type=int, default=1000)
    b.add_argument('--loader-workers', type=int, default=4)
    b.add_argument('--seed', type=int, default=42)
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    (prepare if arguments.command == 'prepare' else train)(arguments)
