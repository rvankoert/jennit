#!/usr/bin/env python3
"""Legacy entry point.

The implementation now lives in the `jennit` package (`jennit.analysis`,
`jennit.cli.analyze`). This file is kept so that
`python flawfinder.py --input_dir ... --output_excel ... --output_dir ...`
keeps working unchanged.
"""
from jennit.cli.analyze import main

if __name__ == "__main__":
    main()
