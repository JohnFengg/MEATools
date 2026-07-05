#!/usr/bin/env python
"""Tests for file utilities."""

import os
import tempfile
import pytest
from meatools.utils.file_utils import (
    find_and_sort_load_dta_files,
    find_cv_subfolders,
    find_and_sort_dta_files_by_candidates
)


class TestFindAndSortLoadDtaFiles:
    """Test DTA file finding and sorting."""

    def test_find_dta_files(self, temp_dir):
        """Test finding DTA files."""
        with open(os.path.join(temp_dir, "test1.cv.DTA"), "w") as f:
            f.write("test")
        with open(os.path.join(temp_dir, "test2.cv.DTA"), "w") as f:
            f.write("test")
        
        result = find_and_sort_load_dta_files(temp_dir, "*cv*.DTA")
        assert len(result) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_sort_by_mtime(self, temp_dir):
        """Test sorting by modification time."""
        import time
        
        file1 = os.path.join(temp_dir, "test1.cv.DTA")
        file2 = os.path.join(temp_dir, "test2.cv.DTA")
        
        with open(file1, "w") as f:
            f.write("test")
        time.sleep(0.1)
        with open(file2, "w") as f:
            f.write("test")
        
        result = find_and_sort_load_dta_files(temp_dir, "*cv*.DTA")
        assert result[0][1] == file1
        assert result[1][1] == file2

    def test_no_files(self, temp_dir):
        """Test with no matching files."""
        result = find_and_sort_load_dta_files(temp_dir, "*.nonexistent")
        assert len(result) == 0


class TestFindCvSubfolders:
    """Test CV subfolder discovery."""

    def test_find_subfolders(self, temp_dir):
        """Test finding subfolders with CV files."""
        subdir = os.path.join(temp_dir, "ECSA", "subdir")
        os.makedirs(subdir)
        
        with open(os.path.join(subdir, "test.cv.DTA"), "w") as f:
            f.write("test")
        
        result = find_cv_subfolders(temp_dir, "*cv*.DTA")
        assert len(result) > 0

    def test_empty_directory(self, temp_dir):
        """Test with empty directory."""
        result = find_cv_subfolders(temp_dir, "*cv*.DTA")
        assert len(result) == 0


class TestFindAndSortDtaFilesByCandidates:
    """Test finding DTA files by candidate directories."""

    def test_find_by_candidates(self, temp_dir):
        """Test finding files in candidate directories."""
        eis_dir = os.path.join(temp_dir, "EIS")
        os.makedirs(eis_dir)
        
        with open(os.path.join(eis_dir, "test.DTA"), "w") as f:
            f.write("test")
        
        result = find_and_sort_dta_files_by_candidates(temp_dir, ("EIS",))
        assert len(result) == 1

    def test_no_candidates(self, temp_dir):
        """Test with no matching candidates."""
        result = find_and_sort_dta_files_by_candidates(temp_dir, ("NONEXISTENT",))
        assert len(result) == 0
