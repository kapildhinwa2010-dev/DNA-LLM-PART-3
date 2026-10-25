"""PRIORITY 1 driver: real teacher extraction bridge + DNA-only evaluation."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world
from helix import KnowledgeDNA
from teacher_bridge import (PROMPT_VERSION, QwenTeacher, build_probe_grid, build_eval_grid,
                            extract_atoms, score_extraction, build_resolver,
                            build_slot_candidates, normalise)
from evaluate import evaluate

OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(OUT, "artifacts")
N_COUNTRIES = int(os.environ.get("HELIX_N_COUNTRIES", "30"))
N_ELEMENTS = int(os.environ.get("HELIX_N_ELEMENTS", "30"))


def domain_lists():
    countries = []
    for line in world.COUNTRIES.strip().splitlines():
        c, cap, cont = line.split("|")
        countries.append((c, cap, cont))
    elements = []
    for i, line in enumerate(world.ELEMENTS.strip().split("|"), start=1):
        sym, name, cat = line.split(" ", 2)
        elements.append((sym, name, cat, i))
    return countries, elements


def main():
    os.makedirs(ART, exist_ok=True)
    countries, elements = domain_lists()
    c_sub = countries[:N_COUNTRIES]
    e_sub = elements[:N_ELEMENTS]

    # ---------------- gold table ----------------
    gold_atoms = world.domain_a_facts()
    gold_set = set((s, r, o) for (s, r, o) in gold_atoms)
    # entity namespace for resolving teacher answers
    namespace = sorted({s for (s, r, o) in gold_atoms} | {o for (s, r, o) in gold_atoms})
    exact = build_resolver(namespace)

    probes = build_probe_grid(c_sub, e_sub)
    evals = build_eval_grid(c_sub, e_sub)
    print(f"probe grid: {len(probes)} probes | eval grid: {len(evals)} queries", flush=True)

    # ---------------- teacher ----------------
    answers_cache = os.path.join(ART, "teacher_answers.json")
    cache = json.load(open(answers_cache)) if os.path.exists(answers_cache) else {}
    if isinstance(cache, list):        # legacy list format
        cache = {r["question"]: r["answer"] for r in cache}
    todo = [p for p in probes if (PROMPT_VERSION + "|" + p[2]) not in cache]
    print(f"probe grid: {len(probes)} | cached answers: {len(cache)} | to ask: {len(todo)}",
          flush=True)
    if todo:
        t0 = time.time()
        teacher = QwenTeacher()
        print("teacher loaded (%.1fs)" % (time.time() - t0), flush=True)
        t0 = time.time()
        CH = 100
        for i in range(0, len(todo), CH):
            chunk = todo[i:i + CH]
            t1 = time.time()
            answers = teacher.answer_batch([p[2] for p in chunk], max_new_tokens=12,
                                           batch_size=32)
            for p, a in zip(chunk, answers):
                cache[PROMPT_VERSION + "|" + p[2]] = a
            json.dump(cache, open(answers_cache, "w"), indent=1)   # checkpoint
            print(f"  teacher {i + len(chunk)}/{len(todo)} probes "
                  f"({(time.time() - t1) / len(chunk):.2f}s/probe)", flush=True)
        teacher.delete()          # <- teacher is gone from here on
        del teacher
        import gc
        gc.collect()
    raw = [{"rel": p[0], "subject": p[1], "question": p[2],
            "answer": cache.get(PROMPT_VERSION + "|" + p[2], cache.get(p[2], "")),
            "gold": p[3]} for p in probes]

    # ---------------- extraction ----------------
    results = [(r["rel"], r["subject"], r["question"], r["answer"], r["gold"]) for r in raw]
    teacher_hit = sum(1 for (rel, s, q, a, g) in results
                      if normalise(a) == normalise(g))
    teacher_res = sum(1 for (rel, s, q, a, g) in results
                      if normalise(a) and normalise(a) in {normalise(x) for x in namespace})
    slots = build_slot_candidates(gold_atoms)
    atoms, xstats = extract_atoms(results, exact, slot_candidates=slots)
    # recall must be measured against the *probed* subset of the gold table
    probed_subjects = {p[1] for p in probes}
    gold_subset = {(s, r, o) for (s, r, o) in gold_atoms if s in probed_subjects}
    scores = score_extraction(atoms, gold_subset)

    # ---------------- compile DNA from extracted atoms ----------------
    p_teacher = os.path.join(ART, "domain_a_teacher.dna")
    dna_t = KnowledgeDNA()
    dna_t.compile_from_teacher(atoms, aliases=world.ALIASES, clues=world.CLUES,
                               rels=world.REL_SPECS, n_slots=8192, nsym=2, replicas=1)
    dna_t.save(p_teacher)

    p_gold = os.path.join(ART, "domain_a_gold.dna")
    dna_g = KnowledgeDNA()
    dna_g.compile_from_teacher(gold_atoms, aliases=world.ALIASES, clues=world.CLUES,
                               rels=world.REL_SPECS, n_slots=8192, nsym=2, replicas=1)
    dna_g.save(p_gold)

    # ---------------- DNA-only evaluation ----------------
    def run_eval(dna, path, tag):
        qs = [(rel, q, exp) for (rel, subj, q, exp) in evals]
        res = evaluate(dna, qs, path)
        # parse failure rate: did the parser even produce a frame?
        parse_fail = 0
        for (rel, subj, q, exp) in evals:
            p = dna.encode_query(q)
            if p.subject is None or p.relation is None:
                parse_fail += 1
        return res, parse_fail

    res_t, pf_t = run_eval(dna_t, p_teacher, "teacher")
    res_g, pf_g = run_eval(dna_g, p_gold, "gold")

    out = {
        "probes": len(probes),
        "eval_queries": len(evals),
        "teacher": {
            "exact_answer_rate": teacher_hit / max(1, len(results)),
            "answer_resolves_to_entity": teacher_res / max(1, len(results)),
        },
        "extraction": {**scores, **xstats, "n_atoms_extracted": len(atoms)},
        "dna_from_teacher": {
            "bytes": res_t.dna_bytes, "atoms": res_t.n_facts,
            "correct": res_t.n_correct, "queries": res_t.n_queries,
            "accuracy": res_t.n_correct / max(1, res_t.n_queries),
            "parse_failures": pf_t,
            "parse_failure_rate": pf_t / max(1, len(evals)),
        },
        "dna_from_gold": {
            "bytes": res_g.dna_bytes, "atoms": res_g.n_facts,
            "correct": res_g.n_correct, "queries": res_g.n_queries,
            "accuracy": res_g.n_correct / max(1, res_g.n_queries),
            "parse_failures": pf_g,
            "parse_failure_rate": pf_g / max(1, len(evals)),
        },
    }
    out["extraction_loss"] = out["dna_from_gold"]["accuracy"] - out["dna_from_teacher"]["accuracy"]
    json.dump(out, open(os.path.join(OUT, "results_teacher.json"), "w"), indent=2)

    print("\n=== TEACHER EXTRACTION BRIDGE ===")
    print(f"probes answered by teacher      : {len(results)}")
    print(f"teacher exact-answer rate        : {out['teacher']['exact_answer_rate']*100:.1f}%")
    print(f"teacher answer resolves to entity: "
          f"{out['teacher']['answer_resolves_to_entity']*100:.1f}%")
    print(f"extraction  P={scores['precision']:.3f} R={scores['recall']:.3f} "
          f"F1={scores['f1']:.3f}  ({scores['tp']}/{scores['n_gold']} gold tuples)")
    print(f"extracted atoms                  : {len(atoms)}")
    print(f"DNA-from-teacher accuracy        : "
          f"{out['dna_from_teacher']['accuracy']*100:.1f}% "
          f"({out['dna_from_teacher']['correct']}/{out['dna_from_teacher']['queries']}), "
          f"parse-fail {out['dna_from_teacher']['parse_failure_rate']*100:.1f}%")
    print(f"DNA-from-gold accuracy (control) : "
          f"{out['dna_from_gold']['accuracy']*100:.1f}%")
    print(f"EXTRACTION LOSS                  : {out['extraction_loss']*100:.1f} points")
    if scores["missing"]:
        print("missing tuples (first 10)       :", scores["missing"][:10])
    if scores["spurious"]:
        print("spurious tuples (first 10)       :", scores["spurious"][:10])


if __name__ == "__main__":
    main()
