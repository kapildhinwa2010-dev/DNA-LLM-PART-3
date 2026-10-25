# Session-2 experiments — adversarial robustness, size curve, interference, rule induction

All numbers produced by `python3 run_experiments.py` (numpy only, CPU, < 3 min).
Raw JSON: `results_experiments.json`.

---

## P3 — adversarial robustness (200 queries)

The DNA's parser is a rule grammar, so we measured exactly where it breaks.

| category | n | parse failures | correct | answered-but-wrong | verdict |
|---|---|---|---|---|---|
| control (in-grammar) | 32 | 0 | **32** | 0 | ✅ |
| distractor entity ("I was in Rome last year. What is the capital of France?") | 50 | 0 | **50** | 0 | ✅ robust |
| multi-clue ("Is the capital of France Rome or Paris?") | 14 | 0 | **14** | 0 | ✅ robust |
| paraphrase outside the grammar ("France's capital city, please?") | 53 | **53** | 0 | 0 | ❌ parse failure |
| typo ("What is the capitl of France?") | 35 | **35** | 0 | 0 | ❌ parse failure |
| negation ("What is NOT the capital of France?") | 16 | 0 | 0 | **16** | ❌ **confidently wrong** |
| **overall** | **200** | 88 | **96** | 16 | 48 % |

**Honest reading.**

1. *Structured* attacks (distractor entities, multi-clue) are handled perfectly — the
   content-addressed lookup is immune to them, which is a real advantage over embedding
   retrieval.
2. *Lexical* attacks (paraphrase outside the grammar, typos) are **total failures**: the DNA
   abstains (parse failure → unknown), which is the safe failure mode, but it means the
   artefact only answers surface forms its grammar covers. **This is the single biggest
   product gap**: a real deployment needs a fuzzy/entity-linking front end (or a learned
   parser) in front of the lattice.
3. *Negation* is the dangerous one: the DNA answers "Paris" to "What is **NOT** the capital
   of France?" — a **confidently wrong** answer, not an abstention. The rule algebra has no
   negation/exception operator. Mitigation path: (a) an explicit "unsupported query type"
   detector, (b) a negation/exception layer in the rule algebra, (c) never let the DNA
   answer a question whose surface form contains a quantifier/negation it cannot ground.

---

## P4 — size curve (synthetic geography, realistic name lengths)

| concepts | atoms | artifact bytes | bytes/atom | queries | accuracy |
|---|---|---|---|---|---|
| 100 | 150 | 6,349 | 42.3 | 100 | 100 % |
| 500 | 750 | 17,583 | 23.4 | 500 | 100 % |
| 1,000 | 1,502 | 31,651 | 21.1 | 1,002 | 100 % |

* **31.7 KB for 1,000 concepts** — comfortably inside the "≤ 100 KB per 1,000 concepts"
  success criterion, and ~3× smaller than the 1 MB target.
* Bytes/atom falls from 42 → 21 as the fixed overhead (grammar, slot index, namespace
  header) amortises: the asymptotic cost is **≈ 21 B/atom** (≈ 168 bits/atom) for a domain
  with ~1.5 facts per concept.
* Accuracy stays at 100 % across the curve → the slot count scales linearly, no collision
  cliff.

---

## P5 — interference / corruption

### P5b — bit-rot in the packed slot lattice (production mode)

2 % of the bytes of every occupied slot are corrupted (Gaussian, σ proportional to the
corruption fraction).

| corrupted byte fraction | accuracy | abstention rate | **wrong-answer rate** |
|---|---|---|---|
| 0 % | 100 % | 0 % | 0 % |
| 2 % | 30.2 % | 69.8 % | **0 %** |
| 5 % | 22.8 % | 77.3 % | **0 %** |
| 10 % | 17.5 % | 82.5 % | **0 %** |
| 25 % | 16.4 % | 83.6 % | **0 %** |

**This is the strongest result of the session.** The per-slot Reed-Solomon parity converts
*any* corruption into **abstention, never a wrong answer**. The system loses coverage, not
correctness. For a knowledge artifact that is exactly the right trade: "I don't know" is
always available, "wrong" never is.

### P5a — carrier (wave) lattice under carrier noise

The wave lattice stores each fact's payload repetition-coded over 4 carriers, phase-coded
by the entity's deterministic unit-modulus code, and reads it back with a matched filter
plus a 2-way interleaved agreement check.

| carrier noise σ | accuracy | abstention rate | answered-but-wrong |
|---|---|---|---|
| 0 | **100 %** | 0 % | 0 % |
| 5 | 2.7 % | 68.3 % | ~29 % |
| 20 | 0 % | 93.1 % | ~6.8 % |
| 60 | 0 % | 97.9 % | ~2.1 % |
| 150 | 0 % | 100 % | 0 % |

* Clean operation is exact (100 %), and **superposition interference is directly measurable**:
  with cells shared between entities (`superpose=True`) accuracy at σ=0 drops to **98.4 %**
  instead of 100 % — the cross-talk predicted by the complex-random-code analysis
  (`interference ≈ payload/√D`).
* But the carrier mode is **not** safe under noise: ~29 % of the queries it still answered
  are wrong. The payload is an uncoded 16-bit integer, so a ±2 carrier perturbation flips
  the id. Fixing this needs a payload ECC (the repetition code gives only a √4 margin).
* **Conclusion (this is the honest version of the "wave" claim):** the carrier/wave encoding
  buys *measurability* (SNR, interference, physical absence) and costs **~3× the space**
  (36.1 KB vs 12.8 KB for the same 567 facts), while the **packed slot + Reed–Solomon**
  design is the one that is *safe* (zero wrong answers under corruption).
  Superposition does **not** buy density for record-like payloads — it buys interference.

---

## P6 — MDL rule inducer

Given only the atom set (no hand-written rules), the inducer searches for consistent
inverse relations and 2-hop chains and scores them by `support / description-length`:

| discovered rule | kind | support | DL | score |
|---|---|---|---|---|
| `capital_of ↔ country_of` | inverse | 71 | 2 | **35.5** |
| `country_of ∘ capital_of` | chain | 71 | 3 | 23.7 |
| `capital_of ∘ country_of` | chain | 71 | 3 | 23.7 |
| `country_of ∘ continent_of` | chain | 71 | 3 | 23.7 |

The hand-written grammar in `world.py` encodes exactly:
`country_of ← capital_of`, `capital_of_country = country_of . capital_of`,
`continent_of_country = country_of . continent_of`.

**The inducer rediscovers the hand-written rule set from data alone.** This is the
mechanism behind `D_derive`: once rules are *learned* rather than hand-specified, the
derivation factor stops being a hand-tuned constant, and the rule layer can be scaled to
domains we never looked at. It also means `D_derive` can be *predicted* on a new domain
before building anything (see DESIGN.md §9).
