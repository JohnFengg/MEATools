#!/usr/bin/env python
"""File discovery utilities for meatools."""

import os
from glob import glob
from pathlib import Path


def find_and_sort_load_dta_files(root_folder, search_key='*cv*.DTA'):
    """Find and sort DTA files by modification time.

    Args:
        root_folder: Directory to search recursively.
        search_key: Glob pattern for matching files. Default: '*cv*.DTA'.

    Returns:
        List of (mtime, filepath) tuples sorted by mtime.
    """
    files = glob(os.path.join(root_folder, '**', search_key), recursive=True)
    file_info = [(os.path.getmtime(f), f) for f in files]
    file_info.sort(reverse=False)
    return file_info


def find_cv_subfolders(root_dir, search_key='*cv*.DTA', log=None):
    """Find subfolders containing CV DTA files.

    Args:
        root_dir: Root directory to search.
        search_key: Glob pattern for matching files. Default: '*cv*.DTA'.
        log: Optional file-like object for logging.

    Returns:
        Sorted list of folder paths.
    """
    root_path = Path(root_dir).resolve()
    csv_folders = {str(p.parent) for p in root_path.glob('**/' + search_key)}
    csv_folders = sorted(csv_folders)
    if log:
        log.write("Subfolders containing CV DTA files:\n")
        for folder in csv_folders:
            log.write(f"\t{folder}\n")
            log.write("**" * 80 + '\n')
    return csv_folders


def find_and_sort_dta_files_by_candidates(root_folder, candidates=('EIS', 'PEIS', 'GEIS')):
    """Find DTA files inside candidate-named subdirectories.

    Args:
        root_folder: Root directory to search.
        candidates: Tuple of subdirectory names to look for.

    Returns:
        List of (mtime, filepath) tuples sorted by mtime.
    """
    root = Path(root_folder)
    files = []
    for folder in candidates:
        target_dirs = list(root.rglob(folder))
        for d in target_dirs:
            files.extend(d.rglob("*.DTA"))
            files.extend(d.rglob("*.dta"))

    file_info = [(f.stat().st_mtime, str(f)) for f in files if f.stat().st_size > 0]
    file_info.sort(key=lambda x: x[0])
    return file_info
