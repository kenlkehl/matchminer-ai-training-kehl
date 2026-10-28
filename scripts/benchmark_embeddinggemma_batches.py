"""Short real-text forward/backward/optimizer benchmarks for GB300 batch sizing."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from finetune_embedder import load_text_model
from trialspace_embedding_losses import KnownPositiveRankingLoss, CachedEmbeddingLoss, RetainedEmbeddingLoss
from trialspace_prompt_profiles import model_prompts
from retrieval_workspace import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-parquet', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--loss-mini-batch-size', type=int, default=0)
    parser.add_argument('--encoder-mini-batch-size', type=int, default=0)
    parser.add_argument('--encoder-token-budget', type=int, default=8192)
    parser.add_argument('--patient-max-tokens', type=int, default=8192)
    parser.add_argument('--cohort-max-tokens', type=int, default=8192,
                        help='Cap the complete cohort input, including its task prefix, for this benchmark only.')
    parser.add_argument('--stress-max-tokens', action='store_true',
                        help='Stress the full token limit, beyond the longest observed text.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--model', default='google/embeddinggemma-2')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--cache-dir', required=True)
    args = parser.parse_args()
    if args.loss_mini_batch_size and args.encoder_mini_batch_size:
        parser.error('Choose cached or retained chunks, not both')
    if not all(1 <= limit <= 8192 for limit in [args.patient_max_tokens, args.cohort_max_tokens]):
        parser.error('Token limits must be between 1 and 8192')
    result = dict(batch_size=args.batch_size, gradient_checkpointing=args.gradient_checkpointing,
                  loss_mini_batch_size=args.loss_mini_batch_size, stress_max_tokens=args.stress_max_tokens,
                  encoder_mini_batch_size=args.encoder_mini_batch_size, encoder_token_budget=args.encoder_token_budget,
                  patient_max_tokens=args.patient_max_tokens, cohort_max_tokens=args.cohort_max_tokens,
                  max_tokens=8192, steps=[], state='preparing')
    atomic_json(args.output, result)
    frame = pd.read_parquet(args.input_parquet, columns=['patient_summary','this_space','eligibility_result'])
    frame = frame.loc[frame.eligibility_result >= 0].reset_index(drop=True)
    pcode, _ = pd.factorize(frame.patient_summary, sort=False)
    ccode, cohorts = pd.factorize(frame.this_space, sort=False)
    identities = np.column_stack([pcode, ccode])
    positive = np.flatnonzero(frame.eligibility_result.to_numpy() >= 1)
    positive_keys = pcode[positive]*len(cohorts) + ccode[positive]
    model = load_text_model(args.model, 'cuda:0', prompt_profile='colbert_matching',
                            max_tokens=8192, revision=args.revision, cache_dir=args.cache_dir)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    from sentence_transformers.sentence_transformer.losses import CoSENTLoss
    losses = dict(mnrl=KnownPositiveRankingLoss(model, positive_keys, len(cohorts)), cosent=CoSENTLoss(model))
    if args.loss_mini_batch_size:
        losses = {name: CachedEmbeddingLoss(loss, args.loss_mini_batch_size) for name, loss in losses.items()}
    if args.encoder_mini_batch_size:
        losses = {name: RetainedEmbeddingLoss(loss, args.encoder_mini_batch_size, args.encoder_token_budget)
                  for name, loss in losses.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)
    rng = np.random.default_rng(42)
    prompts = model_prompts('colbert_matching')
    longest = {}
    for kind, indices in [('mnrl',positive),('cosent',np.arange(len(frame)))]:
        longest[kind] = [int(frame.iloc[indices][column].str.len().idxmax())
                         for column in ['patient_summary','this_space']]
    total = torch.cuda.get_device_properties(0).total_memory
    try:
        # Warm both losses, then time ordinary batches, then include the longest
        # observed patient/cohort in a batch to test maximum padding pressure.
        for index in range(args.steps+4):
            kind = 'mnrl' if index % 2 == 0 else 'cosent'
            pool = positive if kind == 'mnrl' else len(frame)
            selected = rng.choice(pool, size=args.batch_size, replace=False)
            stress = index >= args.steps+2
            if stress: selected[:2] = longest[kind]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            features, real_tokens = [], []
            for column in ['patient_summary','this_space']:
                texts = frame.iloc[selected][column].tolist()
                if stress and args.stress_max_tokens:
                    texts[:2] = [(text+' ')*128 for text in texts[:2]]
                limit = args.patient_max_tokens if column == 'patient_summary' else args.cohort_max_tokens
                values=model.preprocess(texts, prompt=prompts[column], max_length=limit)
                assert values['input_ids'].shape[1] <= limit
                real_tokens.append(int(values['attention_mask'].sum()))
                features.append({k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in values.items()})
            labels = (torch.tensor(identities[selected],device='cuda:0') if kind=='mnrl' else
                      torch.tensor(frame.iloc[selected].eligibility_result.to_numpy()/2.5-1,device='cuda:0',dtype=torch.float32))
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=losses[kind](features, labels)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if not bool(torch.isfinite(loss)): raise ValueError('Non-finite benchmark loss')
            result['steps'].append(dict(index=index,loss=kind,warmup=index<2,stress=stress,
                seconds=elapsed,pairs_per_second=args.batch_size/elapsed,value=float(loss.detach()),
                tokens=[int(x['input_ids'].shape[1]) for x in features],
                real_tokens=real_tokens,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
            result.update(state='benchmarking',total_memory_gib=total/1024**3)
            atomic_json(args.output,result)
            print(json.dumps(result['steps'][-1]),flush=True)
            del features,labels,loss
        ordinary=[r for r in result['steps'] if not r['warmup'] and not r['stress']]
        result.update(state='complete',pairs_per_second=len(ordinary)*args.batch_size/sum(r['seconds'] for r in ordinary),
                      peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['steps']))
    except torch.cuda.OutOfMemoryError as error:
        result.update(state='oom',error=str(error)[:500], failed_step=index,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    finally:
        ordinary=[r for r in result['steps'] if not r['warmup'] and not r['stress']]
        if ordinary:
            result['ordinary_pairs_per_second'] = len(ordinary)*args.batch_size/sum(r['seconds'] for r in ordinary)
        atomic_json(args.output,result)


if __name__ == '__main__':
    main()
