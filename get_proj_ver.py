#!/usr/bin/env python

"""
Get the project version from `pyproject.toml`,
and write it to the given (Bash) file.
"""

import sys
from importlib.metadata import version

def main():
    proj_name = sys.argv[1]
    file_name = sys.argv[2]
    with open(file_name, "wt") as f:
        f.write(f"export PROJ_VER={version(proj_name)}\n")

if __name__ == "__main__":
    main()
