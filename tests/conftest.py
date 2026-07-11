"""Pytest configuration for glyph-lm tests."""

import sys
import os

# Add parent directory to path so 'glyph' module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
