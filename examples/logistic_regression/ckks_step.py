"""Compatibility import for the training compiler graph builder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.model_compiler.logistic_training import build
