"""Import a one-time validated legacy score export while the labeler is stopped."""
import argparse
import fcntl
from pathlib import Path

from .score_journal import import_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Existing label output directory')
    args = parser.parse_args()
    with (args.output / 'labeling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'Imported {import_archive(args.archive, args.output)} completed pairs', flush=True)


if __name__ == '__main__':
    main()
