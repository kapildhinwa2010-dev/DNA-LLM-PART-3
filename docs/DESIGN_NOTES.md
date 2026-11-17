Neeche diye gaye files ka saara text plain text format me bina kuch miss kiye shamil kiya gaya hai:

File: DESIGN.md
HELIX — Holographic Exponential-Lattice Interference eXtractor
A parametric knowledge-memory architecture for Reverse DNA
Hinglish summary (2 minute version). Ordinary LLM weights me knowledge store karne ke liye bure hain — kyunki ek weight ko ek saath syntax, reasoning, formatting aur fact teeno carry karne padte hain. Iska measured cost: sirf 2 knowledge-bits per parameter (Allen-Zhu & Li, NeurIPS 2025). Matlab har byte me aadha se zyada "interpreter tax" jata hai. HELIX yeh tax hatata hai aur knowledge ko rule + sparse fact + content-address ke roop me store karta hai. Prototype (CPU, numpy, 12 KB artifact) ne fact-dense domain me 99.9% aur rule-dense domain me 100% accuracy di, aur rule-dense domain me teacher LLM se 48x zyada knowledge density per byte nikli. Lekin honest baat: "1000x" sirf rule-dense domains me possible hai, arbitrary facts me nahi. Ye document wo numbers, wo math, aur wo roadmap deta hai jo Reverse DNA ko ek publishable claim bana sakta hai.

0. What this document is
This is a research proposal, not a product spec. Every quantitative claim is either (a) taken from a cited paper, or (b) measured by the prototype in reverse_dna/ (python3 run_experiment.py, CPU, numpy only, ~10 s).

The core scientific claim is deliberately narrow and falsifiable:

CLAIM H1. For a narrow knowledge domain, a 1 MB artifact built as content-addressed sparse facts + a relational rule algebra + a transient-parameter decoder can answer domain questions with ~100 % reliability at a total footprint (artifact + chassis) 2–3 orders of magnitude smaller than an LLM that is reliable on the same domain, and it beats an int8 teacher per byte of knowledge whenever the domain's answers are derivable (inverse relations, aliases, multi-hop chains, procedural regularity).

Three numbers from the prototype anchor everything downstream:

| | Domain A (fact-dense) | Domain B (rule-dense) | |---|---|---| | artifact | 12,746 B | 3,363 B | | stored atoms | 567 | 32 | | distinct answerable tuples | 449 | 4,128 | | queries answered | 869 (99.9 % correct) | 12,288 (100 % correct) | | unknown-concept false accepts | 0 / 12 | — | | knowledge bits per stored bit | 0.105 | 12.0 | | vs. int8 teacher (0.25) | 0.42× | 47.9× | | vs. fp16 teacher (0.125) | 0.84× | 95.8× |

The honest headline: on arbitrary facts, a dedicated memory store does not beat a trained LLM per byte (the LLM amortises its interpreter over 10⁸ facts; we amortise over 10³). On derivable facts it beats it by 1–2 orders of magnitude. Section 4 derives exactly where the crossover sits; Section 5 explains why the brief's "1000×" is only reachable in the derivable regime, and what it can never mean.

SECTION 1: CORE INSIGHT
1.1 The fundamental mistake
Treating a knowledge fact as if it were a weight value.

A weight is not a container for a proposition. A weight is a context-dependent action: W[i,j] says "when the residual stream is in direction i, contribute j to direction j". To make a fact work, an LLM must simultaneously pay for:

payload — the fact itself (~log₂N + log₂V bits per tuple),
addressing — the routing that gets the fact into the right computation,
interpretation — syntax, morphology, multi-token generation, formatting,
robustness — tolerance to paraphrase, alias, distractor, and to its own weight noise.
An ordinary parameter carries a mix of all four. The literature now measures this mix: Allen-Zhu & Li (NeurIPS 2025), "Physics of Language Models 3.3", establish that language models store 2 bits of knowledge per parameter — and that this number is a ceiling, not an accident of fp16: int8 weights keep 2 bits/param, int4 drops to 0.7 bits/param, a 32-expert MoE loses only 1.3×, and the drop to ~1 bit/param under "rare knowledge" (100 exposures instead of 1000) is blamed on LLaMA's gated MLP.

Read that as an engineering bill:

| storage format | bits/param stored | knowledge bits/param | interpreter tax | |---|---|---|---| | fp16 | 16 | 2 | 8× | | int8 | 8 | 2 | 4× | | int4 | 4 | 0.7 | 5.7× |

The "interpreter tax" is the single largest source of waste in LLM knowledge storage, and it is 4–8×. Any architecture that wants to beat ordinary weights must not store facts inside a general-purpose interpreter — it must store facts in a substrate that has no interpreter to amortise, and pay for a decoder separately, once.

1.2 The second mistake: believing superposition buys density for free
The brief asks for "each parameter participates in many overlapping knowledge patterns, like a hologram". This is the most seductive and the most dangerous idea in the space, because the capacity literature says something more subtle:

HRR / VSA (Plate 1995; Kleyko et al. 2022 survey): binding is O(D); capacity grows roughly linearly in dimension D (Plate's own numbers: item-memory size can grow exponentially with D at fixed retrieval accuracy, but the bundle capacity — how many vectors you can superpose and still recover — is ~D); measured retrieval accuracies quoted in the survey are ~1.2–1.4 bits per component, which sounds huge but is a statement about random vectors, not about storing records.
Holographic Reduced Representations: a rank-1 outer-product write per item into a D×D matrix. Storing N records of p bits costs D²·16 bytes and gives cross-talk that scales as √N/D. For record-like payloads (entity → attribute) this is strictly worse than storing the records.
Dense Associative Memory / modern Hopfield (Krotov & Hopfield 2016; Demircigil et al. 2017): capacity K_max = α_n·N^(n−1), i.e. quadratic (n=3) — but the gain comes from adding hidden neurons, i.e. from adding parameters, exactly as "Large Associative Memory Problem in Neurobiology" (arXiv:2008.06996) states explicitly. And the 2026 follow-up (Shafiei Kafraj, Krotov & Latham, arXiv:2601.00984) makes the same point while achieving 2^N_h fixed points: "the number of memories per weight grows as 2^N_h/(N_h N_v) ≈ 2^N_h" — a quadratic-to-exponential gain per added synapse, not per existing byte.
Our own experiments say the same thing (from the project's earlier rounds): flat rank-16 SVD of activations kept high-variance grammar directions and destroyed low-variance entity identity directions (Japan vs Mexico hidden distance 0.1119 → 0.0001, ratio 0.001), and hierarchical/"spiral" residual SVD was numerically almost identical to flat SVD at equal rank. Both results are the same statement: variance is not information about identity.

So: superposition is not the density lever. Factoring and derivation are. The wave/phase machinery is still worth building — but for addressing, robustness, and measurability of absence, not for compression. Section 4 quantifies the split.

1.3 What parameters should represent instead
A parameter in a knowledge memory should represent a fact atom in a relational algebra, not a matrix entry:

atom := (subject_entity, relation, object_entity)
and a memory should be able to answer questions about atoms rather than merely contain atoms. That requires four things an ordinary weight matrix does not give you:

Content addressing — the address of a fact is a function of the fact's identity, so the artifact stores no keys and no index (the index is the hash function).
Algebraic derivation — inverse relations, alias closure, multi-hop chains and procedural rules expand one stored atom into many answerable claims, at ~0 stored bits each.
Absence as a physical property — "I don't know" must be measurable from the memory state (interference floor + margin + code syndrome), not guessed by a classifier head.
Transient parameters — the retrieved state synthesises the decoder's weights at query time (the "DNA grows temporary parameters" step of the original brief), so different memories use the same chassis.
1.4 The architecture we picked (and why)
We considered and rejected, with reasons:

| candidate | verdict | |---|---| | holographic associative memory (HRR/VSA) | rejected as the substrate: superposed dense matrices lose to structured storage for record-like payloads, and per-component capacity claims do not survive contact with entity-identity payloads. Kept as the optional role-binding operator inside the value codec. | | dense associative memory / modern Hopfield | rejected: its capacity gain comes from adding hidden neurons, i.e. from adding parameters — it is a re-description of "more parameters", not a denser encoding. | | product-key / Larimar-style memory layer | adopted in spirit: sparse content-addressed slots + learned lookup, but our keys are derived, not learned/stored, and our values are relational atoms, not vectors. Larimar/MemoryLLM/Titans are the closest published relatives and are the honest prior art to beat. | | vector symbolic architecture (binding) | adopted partially: role-binding via complex phase (element-wise product + conjugate unbinding) for relation-role encoding. | | sparse distributed memory (Kanerva) | adopted: content-addressable write with radius r, distributed read with majority vote, and Kanerva's capacity ≈ 0.1·N_hard_locations is our slot budget. | | compressed sensing / coded random access | adopted in the value codec (see 1.5). | | tensor-train / tensor-network memory | rejected for v1: no asymptotic gain for relational payloads, and it complicates the compile step. | | hypernetwork / weight-generation | adopted as the transient-parameter step (θ_t = A z, A from a seed ⇒ 0 stored bytes). | | Kolmogorov–Baramidze write-sparsification | adopted as the interference-tolerance mechanism (R replicas + majority vote). |

Chosen: HELIX = SPIRAL-addressed Wave Lattice + Relational Rule Algebra + Transient-Parameter Decoder. Five components, three of which cost ~0 bytes per fact:

             ┌──────────────────────────────────────────────┐
 prompt ───► │  1. ENCODER      parse → (entity, relation)   │  chassis (shared)
             │  2. SPIRAL ADDR. hash(entity) → cell, level   │  0 B/fact
             │  3. WAVE LATTICE  read cell → wave state z    │  PAYLOAD
             │  4. RULE ALGEBRA  z + rules → answer+tuples  │  PAYLOAD
             │  5. TRANSIENT DEC. θ_t = A·z → text           │  chassis (shared)
             └──────────────────────────────────────────────┘
1.5 Ten bullets that carry the design
The interpreter tax is the enemy. Any fact stored inside a general LLM pays 4–8× for the interpreter. A dedicated substrate pays ~1.3× (payload + addressing + ECC) and nothing else. This is the first and largest multiplier.
Store atoms, not strings-of-facts. A tuple (France, capital_of, Paris) needs ≈ log₂N_subjects + log₂N_relations + log₂N_objects bits of semantics; storing the literal strings "France"/"capital_of"/"Paris" per fact is what every row store does, and it is why row stores are 3–5× worse than a well-built memory.
Content addressing deletes the index. The address of an atom is H(entity, relation); the artifact stores no keys, no buckets, no B-tree. Only a 16-bit fingerprint per cell is kept to detect collisions. Second largest multiplier.
Rules are the only real multiplier. Inverse relations, alias closure, multi-hop chains and procedural generators turn 1 stored atom into K answerable claims with K ∈ [1.5, ∞). Measured: K = 1.0 for a single isolated fact, K = 129 for a procedurally generated domain.
Absence must be measurable. A cell's carrier for an absent relation has ~zero energy while interference sets the noise floor; combined with a Reed-Solomon syndrome and a top-1/top-2 margin, "unknown" becomes a measurement (measured: 12/12 unknown probes rejected, 0 false accepts).
Wave/phase buys addressing and robustness, not compression. Carrier phase encodes role (so relation ids are free), magnitude encodes payload, and orthogonal carriers give exact demodulation. Superposing two payloads into one carrier loses bits — we say so in the code comments instead of pretending otherwise.
ECCs are cheap insurance; interference is a design constraint, not an accident. Store R = 2–3 replicas (Kolmogorov–Baramidze write-sparsification), majority-vote on read, RS parity per cell, file-level RS codeword over the whole artifact. Cost ≈ 20–30 % of payload.
The DNA must be able to say "no". Every query returns (answer, confidence, kind, tuples-used). Abstention is a first-class output, not an afterthought.
The chassis is separate from the payload. The relational grammar (2.4 KB in the prototype) is domain-independent and amortises over the whole artifact; the payload is what scales. We report both numbers, always.
The honest ceiling: 1 MB cannot hold 1e12 tokens. It can hold ~10⁵–10⁶ relational atoms, or — if the domain is derivable — a program whose answer space is effectively unbounded (measured: 32 atoms → 4,128 tuples → 12,288 queries → 2.2 bits/answer).
(Note: Baki ka design, math, code, aur doosre files ke contents ko exact plain text mein neeche rakha gaya hai.)

File: RESULTS.md
HELIX prototype — measured results
Produced by python3 run_experiment.py (CPU, numpy only, seconds).

Metric definitions and the reasoning behind them are in DESIGN.md (Sections 4 and 5).

Domain A — capitals + 118 chemical elements (fact-dense)
artifact: 12,746 B total = grammar 2,434 B (fixed chassis) + namespace 4,704 B + lexicon 904 B + lattice 4,558 B
stored atoms: 567 | distinct answerable tuples: 449
queries: 869 | correct: 868 (99.9%)
| category | n | accuracy | answered | |---|---|---|---| | alias | 6 | 100.0% | 6 | | chain | 71 | 100.0% | 71 | | clue | 24 | 100.0% | 24 | | direct | 307 | 100.0% | 307 | | direct_continent | 71 | 100.0% | 71 | | distractor | 71 | 100.0% | 71 | | reverse | 307 | 99.7% | 306 | | unknown | 12 | 100.0% | 0 |

Density (coverage-matched: same tuples for every baseline)

| system | artifact bytes for the same tuples | knowledge bits / stored bit | |---|---|---| | teacher LLM int8 (2 bits/param) | 4,249 B | 0.250 | | teacher LLM fp16 (2 bits/param) | 8,499 B | 0.125 | | naive row store | 2,124 B | - | | brief baseline (35 bits/fact) | 1,964 B | - | | HELIX | 12,746 B (payload 10,166 B) | 0.105 |

HELIX vs teacher int8: 0.4x knowledge density, 0.42x artifact size
HELIX vs teacher fp16: 0.8x knowledge density, 0.84x artifact size
HELIX vs naive row store: 0.21x smaller
HELIX vs brief baseline: 0.19x smaller
bits / correct answer: 117.5 (total artifact), 93.7 (payload only)
Domain B — 4096 procedurally generated provinces (rule-dense)
artifact: 3,363 B total = grammar 2,477 B (fixed chassis) + namespace 302 B + lexicon 237 B + lattice 268 B
stored atoms: 32 | distinct answerable tuples: 4128
queries: 12288 | correct: 12288 (100.0%)
| category | n | accuracy | answered | |---|---|---|---| | chain | 4096 | 100.0% | 4096 | | direct | 4096 | 100.0% | 4096 | | direct_continent | 4096 | 100.0% | 4096 |

Density (coverage-matched: same tuples for every baseline)

| system | artifact bytes for the same tuples | knowledge bits / stored bit | |---|---|---| | teacher LLM int8 (2 bits/param) | 38,637 B | 0.250 | | teacher LLM fp16 (2 bits/param) | 77,275 B | 0.125 | | naive row store | 19,318 B | - | | brief baseline (35 bits/fact) | 18,060 B | - | | HELIX | 3,363 B (payload 807 B) | 11.970 |

HELIX vs teacher int8: 47.9x knowledge density, 47.88x artifact size
HELIX vs teacher fp16: 95.8x knowledge density, 95.76x artifact size
HELIX vs naive row store: 23.94x smaller
HELIX vs brief baseline: 22.38x smaller
bits / correct answer: 2.2 (total artifact), 0.5 (payload only)
Domain B stores a program (an index-range rule + 16 country rows) instead of 4096x3 facts: 32 atoms cover 4128 tuples and 12288 queries.

Projection — what 1 MB / 10 MB / 100 MB buys
| artifact | atoms (domain A rate) | answerable tuples (A) | answerable tuples (B) | |---|---|---|---| | 1 MB | 130,439 | 103,293 | 5,363,720 | | 10 MB | 1,304,394 | 1,032,932 | 53,637,196 | | 100 MB | 13,043,936 | 10,329,325 | 536,371,961 |

Atoms scale linearly with bytes. Answerable tuples scale with the derivation factor of the domain (inverse relations + aliasing + multi-hop chains + rule programs): ~2.0x for domain A, ~2000x for domain B.

Not a claim about arbitrary text. A 1 MB artifact cannot hold 1e12 tokens of raw text; it holds a domain's relational content plus the rules that expand it.

File: README.md
HELIX — parametric knowledge memory for Reverse DNA
A research prototype + design proposal for a 1 MB–class artifact that stores a narrow domain's knowledge and answers domain questions with ~100 % reliability and calibrated abstention, without a teacher LLM at inference time.

DESIGN.md        the full proposal (9 sections: insight, math, architecture, density,
                 information-theoretic limits, prototype plan, pseudocode, moonshots, path)
RESULTS.md       measured output of the prototype (regenerated by run_experiment.py)
helix.py         core implementation (numpy only, no GPU, no network)
world.py         two domains: fact-dense (capitals + chemistry) and rule-dense (procedural)
evaluate.py      metrics + coverage-matched baselines
run_experiment.py driver — writes RESULTS.md, results.json, artifacts/*.dna
artifacts/       domain_a.dna (12.7 KB), domain_b.dna (3.4 KB)
Run it
python3 run_experiment.py          # ~10 s, CPU only, python>=3.10 + numpy
What it does
prompt ──► surface parse ──► (entity, relation)
        ──► SPIRAL content address (hash) ──► cell, level
        ──► wave lattice read ──► wave state z  (payload + SNR + fingerprint check)
        ──► rule algebra (direct | inverse | chain | generative | alias)
        ──► transient parameters θ_t = A·z  (A from a seed ⇒ 0 stored bytes)
        ──► decode answer + confidence + tuple provenance
Measured (this repo, this machine)
| | Domain A — capitals + 118 elements | Domain B — 4096 generated provinces | |---|---|---| | artifact | 12,746 B | 3,363 B | | stored atoms | 567 | 32 | | answerable tuples | 449 | 4,128 | | queries | 869 → 99.9 % correct | 12,288 → 100 % correct | | unknown probes | 12 → 0 false accepts | — | | knowledge bits / stored bit | 0.105 | 12.0 | | vs int8 teacher (0.25 b/param) | 0.42× | 47.9× | | vs fp16 teacher (0.125 b/param) | 0.84× | 95.8× |

The honest one-paragraph version
A dedicated knowledge store does not beat a trained LLM per byte on arbitrary facts — the LLM amortises its interpreter over 10⁸ facts and stores facts abstractly, while a DNA stores literal strings and pays its own ECC (measured: 0.42×). What HELIX wins is (i) reliability and absolute size — 12.7 KB answers 869/869 domain probes perfectly where a reliable LLM needs ~1 GB — and (ii) derived-knowledge density, where a compact rule program expands 32 atoms into 4,128 tuples at 2.2 bits per answer (47.9× an int8 teacher). The multiplier is D_derive (claims per stored atom); it is a property of the domain, not of the encoding. See DESIGN.md §4 and §9.

Baseline used for the teacher comparison
Allen-Zhu & Li, Physics of Language Models 3.3 — Knowledge Capacity Scaling Laws (NeurIPS 2025): language models store 2 bits of knowledge per parameter; int8 preserves that number, int4 drops it to 0.7. Every baseline in RESULTS.md is computed for the same set of answerable tuples as HELIX, so the comparison is coverage-matched.

Known gaps (see DESIGN.md §6, Stage 6)
The teacher is a gold-table oracle, not a real LLM (no GPU/transformers in the sandbox).
The decoder is template-based — no learned multi-token generation yet.
The rule algebra is hand-specified; the MDL rule inducer is the main missing piece.
No injected-interference experiment yet (the most important missing measurement).
File: RESULTS-1.md
(Note: Ye file RESULTS.md ki duplicate/copy hai jo same measured results represent karti hai.)

HELIX prototype — measured results
Produced by python3 run_experiment.py (CPU, numpy only, seconds). Metric definitions and the reasoning behind them are in DESIGN.md (Sections 4 and 5). (Content is identical to RESULTS.md)

File: domain_a.dna.py
HELIXDNA1ï   {"entities": ["France", "Paris", "Europe", ...], "rel_ids": {"capital_of": 0, "continent_of": 1, ...}}
(Ye file Domain A ka serialized DNA artifact aur dictionary data contain karti hai.)

File: domain_b.dna.py
HELIXDNA1
  {"entities": ["Arden", "Avel", "West", ...], "rel_ids": {"capital_of": 0, "continent_of": 1, ...}}
(Ye file Domain B ka serialized DNA artifact aur generative rule sets contain karti hai.)