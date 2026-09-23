"""Seed a fresh NeuG database from zero. Old data is never imported."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).parent
subprocess.run([sys.executable, str(here / "install_hooks.py")], check=True)
subprocess.run([sys.executable, str(here / "graph.py"), "seed"], check=True)
