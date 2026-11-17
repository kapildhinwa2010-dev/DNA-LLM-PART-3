"""Evaluation harness: accuracy, unknown rejection, and *usable bits per claim*.

The headline metric is the one used by the knowledge-capacity literature
(Allen-Zhu & Li, "Physics of Language Models 3.3"):

        knowledge_bits_per_stored_bit  =  semantic content of the domain
                                           --------------------------------
                                           bits actually used in the artifact

For a trained LLM this number is 2 bits of knowledge per parameter, so
int8 weights give 0.25 and fp16 gives 0.125.  HELIX is compared against those
two constants, at *identical coverage* (same set of answerable tuples).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from helix import KnowledgeDNA


def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


@dataclass
class EvalResult:
    per_category: dict[str, dict] = field(default_factory=dict)
    n_queries: int = 0
    n_correct: int = 0
    dna_bytes: int = 0
    n_facts: int = 0
    tuples: set = field(default_factory=set)
    breakdown: dict = field(default_factory=dict)

    @property
    def payload_bytes(self) -> int:
        return (self.breakdown.get("namespace", 0) + self.breakdown.get("lexicon", 0)
                + self.breakdown.get("lattice", 0))

    @property
    def bits_per_usable_claim(self) -> float:
        return (self.dna_bytes * 8) / max(1, self.n_correct)

    @property
    def bits_per_stored_fact(self) -> float:
        return (self.dna_bytes * 8) / max(1, self.n_facts)


def evaluate(dna: KnowledgeDNA, queries, path: str | None = None) -> EvalResult:
    per_cat: dict[str, dict] = {}
    n_correct = 0
    tuples: set = set()
    for (cat, prompt, expected) in queries:
        parsed = dna.encode_query(prompt)
        ret = dna.retrieve(parsed)
        theta = dna.generate_dynamic_params(ret.wave)
        out = dna.decode(ret, theta)
        d = per_cat.setdefault(cat, {"n": 0, "ok": 0, "answered": 0})
        d["n"] += 1
        if expected is None:                      # unknown probe
            answered = ret.answer is not None
            d["answered"] += int(answered)
            good = not answered
        else:
            good = ret.answer is not None and norm(ret.answer) == norm(expected)
            d["answered"] += int(ret.answer is not None)
        d["ok"] += int(good)
        n_correct += int(good)
        if good:
            tuples.update(ret.tuples)

    res = EvalResult(per_category=per_cat, n_queries=len(queries), n_correct=n_correct,
                     tuples=tuples)
    res.n_facts = dna.n_facts()
    if path:
        res.dna_bytes = os.path.getsize(path)
    else:
        res.dna_bytes = dna.save("/tmp/_helix_tmp.dna")
    res.breakdown = dna.artifact_breakdown()
    return res


# --------------------------------------------------------------------------
# baselines (coverage matched: same set of answerable tuples)
# --------------------------------------------------------------------------
def tuple_bit_complexity(tuples, n_rel: int) -> float:
    """Bit complexity of a relational tuple, in the style of Allen-Zhu & Li.

    A tuple (s, r, o) needs  log2(#subjects) + log2(#relations) + log2(#objects)
    bits.  (Their Theorem 3.1 lower bound uses exactly this shape.)
    """
    if not tuples:
        return 0.0
    subs = {t[0] for t in tuples}
    rels = {t[1] for t in tuples}
    objs = {t[2] for t in tuples}
    return (math.log2(max(2, len(subs))) + math.log2(max(2, len(rels)))
            + math.log2(max(2, len(objs))))


def naive_row_store_bytes(tuples) -> int:
    """A row-per-tuple store: key strings + value strings + an 8 byte index."""
    total = 0
    for (s, r, o) in tuples:
        total += len(s.encode()) + len(r.encode()) + len(o.encode()) + 8
    return total


def llm_artifact_bytes(knowledge_bits: float, bits_per_param: int = 8) -> int:
    """A trained LM stores ~2 knowledge bits per parameter (int8 keeps 2)."""
    return int(knowledge_bits * (bits_per_param / 2.0) / 8.0)


def brief_baseline_bytes(n_tuples: int, fact_bits: float = 35.0) -> int:
    """The optimistic baseline from the brief: 1 parameter ~ 1 atomic fact."""
    return int(n_tuples * fact_bits / 8.0)
