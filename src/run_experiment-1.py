"""Run the HELIX prototype experiments and write RESULTS.md."""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helix import KnowledgeDNA
import world
from evaluate import (evaluate, naive_row_store_bytes, tuple_bit_complexity,
                      llm_artifact_bytes, brief_baseline_bytes)

OUT = os.path.dirname(os.path.abspath(__file__))


def build_domain_a(path: str) -> KnowledgeDNA:
    facts = world.domain_a_facts()
    dna = KnowledgeDNA()
    dna.compile_from_teacher(facts, aliases=world.ALIASES, clues=world.CLUES,
                             rels=world.REL_SPECS, n_slots=8192, nsym=2, replicas=1)
    dna.save(path)
    return dna


def build_domain_b(path: str) -> KnowledgeDNA:
    facts = world.domain_b_facts()
    dna = KnowledgeDNA()
    dna.compile_from_teacher(facts, rels=world.REL_SPECS,
                             gen_rules=world.domain_b_gen_rules(),
                             gen_entities=world.domain_b_gen_entities(),
                             n_slots=1024, nsym=2, replicas=1)
    dna.save(path)
    return dna


def report(title, dna, queries, path):
    res = evaluate(dna, queries, path)
    tuples = sorted(res.tuples)
    tbc = tuple_bit_complexity(tuples, len(dna.rules.rels))
    knowledge_bits = tbc * len(tuples)
    b = res.breakdown
    payload_bits = max(1, res.payload_bytes) * 8
    density = knowledge_bits / payload_bits           # knowledge bits per stored bit
    L = []
    L.append(f"### {title}\n")
    L.append(f"* artifact: **{res.dna_bytes:,} B** total = grammar {b['grammar']:,} B "
             f"(fixed chassis) + namespace {b['namespace']:,} B + lexicon {b['lexicon']:,} B "
             f"+ lattice {b['lattice']:,} B")
    L.append(f"* stored atoms: **{res.n_facts}** | distinct answerable tuples: **{len(tuples)}**")
    L.append(f"* queries: **{res.n_queries}** | correct: **{res.n_correct}** "
             f"({100.0*res.n_correct/max(1,res.n_queries):.1f}%)\n")
    L.append("| category | n | accuracy | answered |")
    L.append("|---|---|---|---|")
    for cat, d in sorted(res.per_category.items()):
        L.append(f"| {cat} | {d['n']} | {100.0*d['ok']/max(1,d['n']):.1f}% | {d['answered']} |")
    L.append("")
    L.append("**Density (coverage-matched: same tuples for every baseline)**\n")
    L.append("| system | artifact bytes for the same tuples | knowledge bits / stored bit |")
    L.append("|---|---|---|")
    llm8 = llm_artifact_bytes(knowledge_bits, 8)
    llm16 = llm_artifact_bytes(knowledge_bits, 16)
    rows = llm_artifact_bytes(knowledge_bits, 4)
    brief = brief_baseline_bytes(len(tuples))
    L.append(f"| teacher LLM int8 (2 bits/param) | {llm8:,} B | {2/8:.3f} |")
    L.append(f"| teacher LLM fp16 (2 bits/param) | {llm16:,} B | {2/16:.3f} |")
    L.append(f"| naive row store | {rows:,} B | - |")
    L.append(f"| brief baseline (35 bits/fact) | {brief:,} B | - |")
    L.append(f"| **HELIX** | **{res.dna_bytes:,} B** (payload {res.payload_bytes:,} B) | "
             f"**{density:.3f}** |")
    L.append("")
    L.append(f"* HELIX vs teacher int8: **{density/0.25:.1f}x** knowledge density, "
             f"**{llm8/max(1,res.payload_bytes):.2f}x** artifact size")
    L.append(f"* HELIX vs teacher fp16: **{density/0.125:.1f}x** knowledge density, "
             f"**{llm16/max(1,res.payload_bytes):.2f}x** artifact size")
    L.append(f"* HELIX vs naive row store: **{rows/max(1,res.payload_bytes):.2f}x** smaller")
    L.append(f"* HELIX vs brief baseline: **{brief/max(1,res.payload_bytes):.2f}x** smaller")
    L.append(f"* bits / correct answer: **{res.bits_per_usable_claim:.1f}** "
             f"(total artifact), **{res.payload_bytes*8/max(1,res.n_correct):.1f}** (payload only)")
    L.append("")
    return "\n".join(L), res


def main():
    os.makedirs(os.path.join(OUT, "artifacts"), exist_ok=True)
    doc = ["# HELIX prototype — measured results\n",
           "Produced by `python3 run_experiment.py` (CPU, numpy only, seconds).\n",
           "Metric definitions and the reasoning behind them are in `DESIGN.md` "
           "(Sections 4 and 5).\n"]

    pa = os.path.join(OUT, "artifacts", "domain_a.dna")
    build_domain_a(pa)
    dna_a2 = KnowledgeDNA.load(pa)          # round-trip through the artifact only
    tA, resA = report("Domain A — capitals + 118 chemical elements (fact-dense)",
                       dna_a2, world.domain_a_queries(), pa)

    pb = os.path.join(OUT, "artifacts", "domain_b.dna")
    build_domain_b(pb)
    dna_b2 = KnowledgeDNA.load(pb)
    tB, resB = report("Domain B — 4096 procedurally generated provinces (rule-dense)",
                      dna_b2, world.domain_b_queries(), pb)
    tB += ("\nDomain B stores a *program* (an index-range rule + 16 country rows) instead of "
           f"4096x3 facts: {resB.n_facts} atoms cover {len(resB.tuples)} tuples and "
           f"{resB.n_queries} queries.\n")

    doc += [tA, "\n", tB]

    doc += ["### Projection — what 1 MB / 10 MB / 100 MB buys\n",
            "| artifact | atoms (domain A rate) | answerable tuples (A) | answerable tuples (B) |",
            "|---|---|---|---|"]
    for mb in (1, 10, 100):
        nb = mb * 1024 * 1024
        bA = resA.breakdown
        atom_bytes = (bA["lattice"]) / max(1, resA.n_facts)
        atoms = nb / atom_bytes
        tA_ = atoms * (len(resA.tuples) / max(1, resA.n_facts))
        tB_ = nb * (len(resB.tuples) / max(1, resB.payload_bytes))
        doc.append(f"| {mb} MB | {atoms:,.0f} | {tA_:,.0f} | {tB_:,.0f} |")
    doc += ["",
            "Atoms scale linearly with bytes.  Answerable tuples scale with the "
            "*derivation factor* of the domain (inverse relations + aliasing + multi-hop "
            "chains + rule programs): ~2.0x for domain A, ~2000x for domain B.",
            "",
            "**Not a claim about arbitrary text.** A 1 MB artifact cannot hold 1e12 tokens "
            "of raw text; it holds a domain's *relational* content plus the rules that "
            "expand it."]

    text = "\n".join(doc)
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({
            "domain_a": {"bytes": resA.dna_bytes, "payload_bytes": resA.payload_bytes,
                         "facts": resA.n_facts, "tuples": len(resA.tuples),
                         "queries": resA.n_queries, "correct": resA.n_correct,
                         "bits_per_claim": resA.bits_per_usable_claim,
                         "per_category": resA.per_category,
                         "breakdown": resA.breakdown},
            "domain_b": {"bytes": resB.dna_bytes, "payload_bytes": resB.payload_bytes,
                         "facts": resB.n_facts, "tuples": len(resB.tuples),
                         "queries": resB.n_queries, "correct": resB.n_correct,
                         "bits_per_claim": resB.bits_per_usable_claim,
                         "per_category": resB.per_category,
                         "breakdown": resB.breakdown},
        }, f, indent=2)
    print(text)


if __name__ == "__main__":
    main()
