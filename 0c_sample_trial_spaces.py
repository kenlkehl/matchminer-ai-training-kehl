#!/usr/bin/env python3
"""Sample complete trials, retaining every space, with the DFCI evaluation holdout."""
import argparse
import json
from pathlib import Path

import pandas as pd
from trial_space_format import has_required_space_fields

def clean_trial_spaces(line_item_space_dataframe):
    return line_item_space_dataframe.this_space.apply(has_required_space_fields)


def dfci_nct_ids_open_during(dfci_trials, first_year, last_year):
    """NCT IDs whose dates do not place them wholly outside first_year..last_year.

    The dates are ClinicalTrials.gov study start/completion dates at month or day
    precision, so comparing years is exact. A missing date leaves that side open,
    and a trial still open to accrual in the dump counts as open.
    """
    dfci_trials = dfci_trials.dropna(subset=['nct_id'])
    for column in ['open_date', 'close_date']:
        dates = dfci_trials[column].dropna()
        assert dates.str.fullmatch(r'\d{4}-\d{2}(-\d{2})?').all(), f'unexpected {column} format'
    open_year = pd.to_numeric(dfci_trials.open_date.str[:4])
    close_year = pd.to_numeric(dfci_trials.close_date.str[:4])
    outside_window = (close_year < first_year) | (open_year > last_year)
    open_during = ~outside_window | (dfci_trials.status == 'OPEN TO ACCRUAL')
    return set(dfci_trials.nct_id[open_during])


def sample_trial_spaces(input_csv, output_csv, dfci_csv, *, n_trials=3000, seed=42,
                        first_year=2025, last_year=2026, chunksize=2000):
    dfci_trials = pd.read_csv(dfci_csv)
    excluded = dfci_nct_ids_open_during(dfci_trials, first_year, last_year)
    # Scan only the small columns first; the full extraction CSV can contain
    # repeated multi-megabyte reasoning traces. Sampling is by trial, not space.
    counts, invalid = {}, set()
    for chunk in pd.read_csv(input_csv, usecols=['nct_id', 'this_space'], chunksize=chunksize):
        good = clean_trial_spaces(chunk)
        invalid.update(chunk.loc[~good, 'nct_id'])
        for trial, count in chunk.groupby('nct_id').size().items():
            counts[trial] = counts.get(trial, 0) + int(count)
    eligible = sorted(set(counts) - excluded - invalid)
    if len(eligible) < n_trials:
        raise ValueError(f'Only {len(eligible)} complete, non-held-out trials; need {n_trials}')
    sampled = pd.Series(eligible).sample(n=n_trials, random_state=seed).tolist()
    order = {trial: i for i, trial in enumerate(sampled)}
    frames = []
    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        selected = chunk.loc[chunk.nct_id.isin(order)].copy()
        if not selected.empty:
            frames.append(selected)
    output = pd.concat(frames, ignore_index=True)
    output['_sample_order'] = output.nct_id.map(order)
    output = output.sort_values('_sample_order', kind='stable').drop(columns='_sample_order').reset_index(drop=True)
    output['space_index'] = output.index
    assert output.groupby('nct_id').size().to_dict() == {trial: counts[trial] for trial in sampled}
    assert clean_trial_spaces(output).all()
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    manifest = dict(input=str(Path(input_csv).resolve()), output=str(path.resolve()),
                    dfci_holdout=str(Path(dfci_csv).resolve()), first_year=first_year, last_year=last_year,
                    seed=seed, sampled_trials=n_trials, sampled_spaces=len(output),
                    source_trials=len(counts), eligible_trials=len(eligible),
                    excluded_dfci_trials=sorted(set(counts) & excluded),
                    invalid_trials=sorted(invalid), sampled_nct_ids=sampled,
                    all_spaces_retained=True)
    receipt = path.with_suffix('.sampling.json')
    receipt.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', default='../data/no_phi/trial_space_lineitems.csv')
    parser.add_argument('--output-csv', default='../data/no_phi/sample_trial_space_lineitems.csv')
    parser.add_argument('--dfci-trials-csv', default='../data/no_phi/dfci_trials_summary_10-18-26.csv')
    parser.add_argument('--n-trials', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--first-year', type=int, default=2025)
    parser.add_argument('--last-year', type=int, default=2026)
    parser.add_argument('--chunksize', type=int, default=2000)
    args = parser.parse_args()
    sample_trial_spaces(args.input_csv, args.output_csv, args.dfci_trials_csv,
                        n_trials=args.n_trials, seed=args.seed, first_year=args.first_year,
                        last_year=args.last_year, chunksize=args.chunksize)


if __name__ == '__main__':
    main()
