"""Seed a fresh NeuG database from zero. Old data is never imported."""
import subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, str(Path(__file__).parent / "graph.py"), "seed"], check=True)
