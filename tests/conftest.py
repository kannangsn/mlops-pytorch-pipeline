import sys
from pathlib import Path

# Make src/ importable without packaging the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
