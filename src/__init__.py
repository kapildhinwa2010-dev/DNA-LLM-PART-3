"""
HELIX - Holographic Exponential-Lattice Interference eXtractor
A parametric knowledge-memory architecture for Reverse DNA.

This package contains the core implementation of the HELIX system.
"""

from .helix import KnowledgeDNA, RetrievalResult, RelSpec
from .helix_carrier import CarrierLattice
from .world import REL_SPECS, ALIASES, CLUES, COUNTRIES, ELEMENTS
from .tokenizer import QwenTokenizer, chatml_ids
from .evolve import evaluate, EvalResult

__version__ = "0.1.0"
__all__ = [
    'KnowledgeDNA',
    'RetrievalResult',
    'RelSpec',
    'CarrierLattice',
    'REL_SPECS',
    'ALIASES',
    'CLUES',
    'COUNTRIES',
    'ELEMENTS',
    'QwenTokenizer',
    'chatml_ids',
    'evaluate',
    'EvalResult',
]
