#!/usr/bin/env python3
"""Prune checkpoints under results/*/checkpoints keeping the best 3 by val_wer (lower is better)
and always keep 'last.ckpt'. Deletes files permanently.
"""
import os, re, shutil, argparse
from pathlib import Path

def find_ckpts(root):
    for p in Path(root).rglob('checkpoints'):
        if p.is_dir():
            files = [f for f in p.iterdir() if f.is_file() and f.suffix=='.ckpt']
            if files:
                yield p, files

re_val = re.compile(r'val_wer=([0-9]+(?:\.[0-9]+)?)')


def val_from_name(name):
    m = re_val.search(name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def prune_dir(ckpt_dir, files, keep=3):
    # Keep last.ckpt and top-k by val_wer ascending. If val_wer missing, use mtime ascending.
    last = [f for f in files if f.name == 'last.ckpt']
    others = [f for f in files if f.name != 'last.ckpt']

    # Annotate with val or mtime
    annotated = []
    for f in others:
        v = val_from_name(f.name)
        if v is None:
            v = f.stat().st_mtime
        annotated.append((f, v))

    # Sort by v (lower is better for val_wer, mtime for missing)
    annotated.sort(key=lambda x: x[1])

    keep_set = set()
    for i, (f, v) in enumerate(annotated[:keep]):
        keep_set.add(f)

    for f in last:
        keep_set.add(f)

    to_delete = [f for f in files if f not in keep_set]

    return list(keep_set), to_delete


def main(root='results', keep=3, dry_run=True):
    total_deleted = 0
    total_freed = 0
    actions = []
    for ckpt_dir, files in find_ckpts(root):
        keep_set, to_delete = prune_dir(ckpt_dir, files, keep=keep)
        if not to_delete:
            continue
        actions.append((ckpt_dir, keep_set, to_delete))

    print(f"Found {len(actions)} checkpoint directories with deletable files")
    for ckpt_dir, keep_set, to_delete in actions:
        print(f"\nDirectory: {ckpt_dir}")
        print("Keeping:")
        for k in sorted(keep_set):
            print("  ", k.name)
        print("Deleting:")
        for d in sorted(to_delete):
            sz = d.stat().st_size
            print(f"  {d.name}  ({sz/1024/1024:.1f} MB)")

    if dry_run:
        print('\nDry run enabled; no files were removed. Re-run with --doit to delete files.')
        return 0

    # Confirm
    confirm = input('\nProceed to delete the listed files? Type YES to confirm: ')
    if confirm != 'YES':
        print('Aborted by user')
        return 1

    # Delete
    for ckpt_dir, keep_set, to_delete in actions:
        for d in to_delete:
            try:
                sz = d.stat().st_size
                d.unlink()
                total_deleted += 1
                total_freed += sz
                print(f'Deleted {d} ({sz/1024/1024:.1f} MB)')
            except Exception as e:
                print(f'Failed to delete {d}: {e}')

    print(f'Finished. Deleted {total_deleted} files; freed {total_freed/1024/1024:.1f} MB')
    return 0

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results')
    parser.add_argument('--keep', type=int, default=3)
    parser.add_argument('--doit', action='store_true')
    args = parser.parse_args()
    sys.exit(main(root=args.root, keep=args.keep, dry_run=not args.doit))
