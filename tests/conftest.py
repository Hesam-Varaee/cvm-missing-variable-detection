"""Pytest configuration: makes src/ importable without modifying any
production code or its (intentionally flat, sibling-style) imports."""
import sys
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
