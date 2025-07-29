#!/usr/bin/env python3
"""
Quick launcher for LQMF application
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cli.main import main

if __name__ == "__main__":
    main()