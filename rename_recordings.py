#!/usr/bin/env python3
"""Rename recordings using their existing number; no audio conversion.

Preview: python3 rename_recordings.py /path/to/folder en 04
Apply:   python3 rename_recordings.py /path/to/folder en 04 --apply

Only immediate audio files are processed, not subfolders or symlinks.
Names with multiple numbers are skipped unless already in the output format.
"""

import argparse
import os
from pathlib import Path
import re
import sys

AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg',
                    '.opus', '.aiff', '.aif', '.wma', '.amr', '.mp4', '.3gp'}


def speaker_id(value):
    match = re.fullmatch(r'(?:spk)?([0-9]+)', value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError('Use a numeric speaker ID: 4, 04, or spk04.')
    return f'spk{int(match.group(1)):02d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path)
    parser.add_argument('language', choices=['en', 'ne'])
    parser.add_argument('speaker', type=speaker_id)
    parser.add_argument('--apply', action='store_true', help='Apply the previewed renames.')
    args = parser.parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f'Not a directory: {folder}')

    plan = []
    targets = {}
    errors = []
    for source in sorted(folder.iterdir()):
        if source.is_symlink() or not source.is_file():
            continue
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        formatted = re.fullmatch(r'(?:en|nep)_spk[0-9]+_([0-9]+)', source.stem)
        numbers = [formatted.group(1)] if formatted else re.findall(r'[0-9]+', source.stem)
        if len(numbers) != 1:
            print(f'SKIP {source.name!r}: expected one number, found {len(numbers)}.')
            continue
        number = int(numbers[0])
        target = source.with_name(f'{args.language}_{args.speaker}_{number:03d}{source.suffix}')
        if source == target:
            print(f'OK   {source.name!r}: already named correctly.')
            continue
        key = target.name.casefold()
        if key in targets:
            errors.append(f'{source.name!r} and {targets[key].name!r} both map to {target.name!r}.')
        targets[key] = source
        if os.path.lexists(target):
            errors.append(f'Destination already exists: {target.name!r}.')
        plan.append((source, target))

    for source, target in plan:
        print(f'{source.name!r} -> {target.name!r}')
    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        print('No files changed. Resolve the conflicts and retry.', file=sys.stderr)
        return 1
    if not args.apply:
        print(f'Preview: {len(plan)} file(s). Add --apply to rename.')
        return 0

    completed = 0
    for source, target in plan:
        try:
            # Creating a hard link fails if the target exists: never overwrite.
            # Then remove the old name. Audio bytes and extension stay unchanged.
            os.link(source, target)
            try:
                source.unlink()
            except OSError:
                target.unlink()
                raise
        except OSError as error:
            print(f'ERROR on {source.name!r}: {error}. Stopped after {completed} rename(s).',
                  file=sys.stderr)
            return 1
        completed += 1
    print(f'Renamed {completed} file(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
