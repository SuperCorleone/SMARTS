#!/usr/bin/env python3
"""Compatibility wrapper for the old RQ4 entry point.

The drift experiment is now canonical RQ1. This wrapper forwards to
`run_rq1.py` so older commands keep working during the transition.
"""

from run_rq1 import main


if __name__ == '__main__':
    main()
