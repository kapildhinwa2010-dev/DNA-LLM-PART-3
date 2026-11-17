# HELIX — Holographic Exponential-Lattice Interference eXtractor
### A parametric knowledge-memory architecture for *Reverse DNA*

> **Quick Summary (2 minute version).** Ordinary LLM weights are inefficient for
> knowledge storage because each weight must simultaneously encode syntax, reasoning, formatting **and** facts.
> The measured cost: **only 2 knowledge-bits per parameter** (Allen-Zhu & Li, NeurIPS 2025).
> This means more than half of each byte goes to "interpreter tax". HELIX eliminates this tax by storing
> knowledge as *rule + sparse fact + content-address*. The prototype (CPU, numpy, 12 KB artifact) achieves
> **99.9% accuracy in fact-dense domains** and **100% accuracy in rule-dense domains**, with **48x better
> knowledge density per byte** than the teacher LLM in rule-dense domains. However, the honest truth is:
> **"1000x" is only possible in rule-dense domains**, not for arbitrary facts. This document provides the
> numbers, the math, and the roadmap that could make Reverse DNA a publishable claim.

---

## 0. What this document is

This is a **research proposal**, not a product spec. Every quantitative claim is either
(a) taken from a cited paper, or (b) measured by the prototype in `reverse_dna/`
(`python3 run_experiment.py`, CPU, numpy only, ~10 s).

The core scientific claim is deliberately narrow and falsifiable:

> **CLAIM H1.** For a *narrow* knowledge domain, a 1 MB artifact built as
> **content-addressed sparse facts + a relational rule algebra + a transient-parameter
> decoder** can answer *domain* questions with ~100 % reliability at a total footprint
> (artifact + chassis) 2–3 orders of magnitude smaller than an LLM that is reliable on the
> same domain, and it beats an int8 teacher *per byte of knowledge* whenever the domain's
> answers are **derivable** (inverse relations, aliases, multi-hop chains, procedural
> regularity).

Three numbers from the prototype anchor everything downstream:

| | Domain A (fact-dense) | Domain B (rule-dense) |
|---|---|---|
| artifact | 12,746 B | 3,363 B |
| stored atoms | 567 | 32 |
| distinct answerable tuples | 449 | 4,128 |
| queries answered | 869 (99.9 % correct) | 12,288 (100 % correct) |
| unknown-concept false accepts | 0 / 12 | — |
| **knowledge bits per stored bit** | 0.105 | **12.0** |
| **vs. int8 teacher (0.25)** | **0.42×** | **47.9×** |
| **vs. fp16 teacher (0.125)** | 0.84× | 95.8× |

The honest headline: **on arbitrary facts, a dedicated memory store does *not* beat a trained
LLM per byte** (the LLM amortises its interpreter over 10⁸ facts; we amortise over 10³).
**On derivable facts it beats it by 1–2 orders of magnitude.** Section 4 derives exactly where
the crossover sits; Section 5 explains why the brief's "1000×" is only reachable in the
derivable regime, and what it can never mean.

---

## SECTION 1: CORE INSIGHT

### 1.1 The fundamental mistake

> **Treating a knowledge fact as if it were a weight value.**

A weight is not a container for a proposition. A weight is a *context-dependent action*:
`W[i,j]` says "when the residual stream is in direction *i*, contribute *j* to direction *j*".
To make a fact work, an LLM must simultaneously pay for:

1. **payload** — the fact itself (~log₂N + log₂V bits per tuple),
2. **addressing** — the routing that gets the fact into the right computation,
3. **interpretation** — syntax, morphology, multi-token generation, formatting,
4. **robustness** — tolerance to paraphrase, alias, distractor, and to its own weight noise.

An ordinary parameter carries a *mix* of all four. The literature now measures this mix:
**Allen-Zhu & Li (NeurIPS 2025), "Physics of Language Models 3.3", establish that language
models store 2 bits of knowledge per parameter** — and that this number is a *ceiling*, not an
accident of fp16: int8 weights keep 2 bits/param, **int4 drops to 0.7 bits/param**, a 32-expert
MoE loses only 1.3×, and the drop to ~1 bit/param under "rare knowledge" (100 exposures instead
of 1000) is blamed on LLaMA's *gated MLP*.

Read that as an engineering bill:

| storage format | bits/param stored | knowledge bits/param | **interpreter tax** |
|---|---|---|---|
| fp16 | 16 | 2 | **8×** |
| int8 | 8 | 2 | **4×** |
| int4 | 4 | 0.7 | **5.7×** |

**The "interpreter tax" is the single largest source of waste in LLM knowledge storage, and it
is 4–8×.** Any architecture that wants to beat ordinary weights must *not* store facts inside a
general-purpose interpreter — it must store facts in a substrate that has **no interpreter to
amortise**, and pay for a decoder separately, once.

### 1.2 The second mistake: believing superposition buys density for free

The brief asks for "each parameter participates in many overlapping knowledge patterns, like a
hologram". This is the most seductive and the most *dangerous* idea in the space, because the
capacity literature says something more subtle:

* **HRR / VSA** (Plate 1995; Kleyko et al. 2022 survey): binding is O(D); capacity grows
  roughly *linearly* in dimension D (Plate's own numbers: item-memory size can grow
  *exponentially* with D at fixed retrieval accuracy, but the *bundle* capacity — how many
  vectors you can superpose and still recover — is ~D); measured retrieval accuracies quoted in
  the survey are ~1.2–1.4 bits *per component*, which sounds huge but is a statement about
  *random* vectors, not about storing records.
* **Holographic Reduced Representations**: a rank-1 outer-product write per item into a D×D
  matrix. Storing N records of p bits costs D²·16 bytes and gives cross-talk that scales as
  √N/D. For record-like payloads (entity → attribute) this is **strictly worse** than storing
  the records.
* **Dense Associative Memory / modern Hopfield** (Krotov & Hopfield 2016; Demircigil et al.
  2017): capacity K_max = α_n·N^(n−1), i.e. quadratic (n=3) — but the *gain comes from adding
  hidden neurons*, i.e. from **adding parameters**, exactly as
  "Large Associative Memory Problem in Neurobiology" (arXiv:2008.06996) states explicitly.
  And the 2026 follow-up (Shafiei Kafraj, Krotov & Latham, arXiv:2601.00984) makes the same
  point while achieving 2^N_h fixed points: *"the number of memories per weight grows as
  2^N_h/(N_h N_v) ≈ 2^N_h"* — a quadratic-to-exponential gain **per added synapse**, not per
  existing byte.

**Our own experiments say the same thing** (from the project's earlier rounds): flat rank-16
SVD of activations kept high-variance grammar directions and destroyed low-variance *entity
identity* directions (Japan vs Mexico hidden distance 0.1119 → 0.0001, ratio 0.001), and
hierarchical/"spiral" residual SVD was numerically almost identical to flat SVD at equal rank.
Both results are the *same* statement: **variance is not information about identity.**

So: **superposition is not the density lever. Factoring and derivation are.** The wave/phase
machinery is still worth building — but for *addressing, robustness, and measurability of
absence*, not for compression. Section 4 quantifies the split.

### 1.3 What parameters should represent instead

A parameter in a knowledge memory should represent a **fact atom in a relational algebra**, not
a matrix entry:

atom := (subject_entity, relation, object_entity)and a memory should be able to answer **questions about atoms** rather than merely *contain*
atoms. That requires four things an ordinary weight matrix does not give you:

1. **Content addressing** — the address of a fact is a *function of the fact's identity*, so the
   artifact stores **no keys and no index** (the index is the hash function).
2. **Algebraic derivation** — inverse relations, alias closure, multi-hop chains and procedural
   rules expand one stored atom into many answerable claims, at ~0 stored bits each.
3. **Absence as a physical property** — "I don't know" must be measurable from the memory
   state (interference floor + margin + code syndrome), not guessed by a classifier head.
4. **Transient parameters** — the retrieved state *synthesises* the decoder's weights at query
   time (the "DNA grows temporary parameters" step of the original brief), so different memories
   use the same chassis.

### 1.4 The architecture we picked (and why)

We considered and rejected, with reasons:

| candidate | verdict |
|---|---|
| holographic associative memory (HRR/VSA) | rejected as the *substrate*: superposed dense matrices lose to structured storage for record-like payloads, and per-component capacity claims do not survive contact with entity-identity payloads. **Kept as the optional role-binding operator** inside the value codec. |
| dense associative memory / modern Hopfield | rejected: its capacity gain comes from *adding hidden neurons*, i.e. from adding parameters — it is a re-description of "more parameters", not a denser encoding. |
| product-key / Larimar-style memory layer | **adopted in spirit**: sparse content-addressed slots + learned lookup, but our keys are *derived*, not learned/stored, and our values are relational atoms, not vectors. Larimar/MemoryLLM/Titans are the closest published relatives and are the honest prior art to beat. |
| vector symbolic architecture (binding) | **adopted partially**: role-binding via complex phase (element-wise product + conjugate unbinding) for relation-role encoding. |
| sparse distributed memory (Kanerva) | **adopted**: content-addressable write with radius r, distributed read with majority vote, and Kanerva's capacity ≈ 0.1·N_hard_locations is our slot budget. |
| compressed sensing / coded random access | **adopted in the value codec** (see 1.5). |
| tensor-train / tensor-network memory | rejected for v1: no asymptotic gain for *relational* payloads, and it complicates the compile step. |
| hypernetwork / weight-generation | **adopted** as the transient-parameter step (θ_t = A z, A from a seed ⇒ 0 stored bytes). |
| Kolmogorov–Baramidze write-sparsification | **adopted** as the interference-tolerance mechanism (R replicas + majority vote). |

**Chosen: HELIX = SPIRAL-addressed Wave Lattice + Relational Rule Algebra + Transient-Parameter
Decoder.** Five components, three──────────────────────────────────────────────┐
prompt ───► │  1. ENCODER      parse → (entity, relation)   │  chassis (shared)
│  2. SPIRAL ADDR. hash(entity) → cell, level   │  0 B/fact
│  3. WAVE LATTICE  read cell → wave state z    │  PAYLOAD
│  4. RULE ALGEBRA  z + rules → answer+tuples  │  PAYLOAD
│  5. TRANSIENT DEC. θ_t = A·z → text           │  chassis (shared)
└──────────────────────────────────────────────┘## 1.5 Ten bullets that carry the design

1. **The interpreter tax is the enemy.** Any fact stored inside a general LLM pays 4–8× for
   the interpreter. A dedicated substrate pays ~1.3× (payload + addressing + ECC) and nothing
   else. *This is the first and largest multiplier.*
2. **Store atoms, not strings-of-facts.** A tuple `(France, capital_of, Paris)` needs
   ≈ log₂N_subjects + log₂N_relations + log₂N_objects bits *of semantics*; storing the literal
   strings "France"/"capital_of"/"Paris" per fact is what every row store does, and it is why
   row stores are 3–5× worse than a well-built memory.
3. **Content addressing deletes the index.** The address of an atom is `H(entity, relation)`;
   the artifact stores no keys, no buckets, no B-tree. Only a 16-bit fingerprint per cell is
   kept to detect collisions. *Second largest multiplier.*
4. **Rules are the only real multiplier.** Inverse relations, alias closure, multi-hop chains
   and procedural generators turn 1 stored atom into K answerable claims with K ∈ [1.5, ∞).
   Measured: K = 1.0 for a single isolated fact, **K = 129 for a procedurally generated domain**.
5. **Absence must be measurable.** A cell's carrier for an absent relation has ~zero energy
   while interference sets the noise floor; combined with a Reed-Solomon syndrome and a
   top-1/top-2 margin, "unknown" becomes a *measurement* (measured: 12/12 unknown probes
   rejected, 0 false accepts).
6. **Wave/phase buys addressing and robustness, not compression.** Carrier phase encodes
   *role* (so relation ids are free), magnitude encodes payload, and orthogonal carriers give
   exact demodulation. Superposing two payloads into one carrier *loses* bits — we say so in
   the code comments instead of pretending otherwise.
7. **ECCs are cheap insurance; interference is a design constraint, not an accident.** Store
   R = 2–3 replicas (Kolmogorov–Baramidze write-sparsification), majority-vote on read, RS
   parity per cell, file-level RS codeword over the whole artifact. Cost ≈ 20–30 % of payload.
8. **The DNA must be able to say "no".** Every query returns (answer, confidence, kind,
   tuples-used). Abstention is a first-class output, not an afterthought.
9. **The chassis is separate from the payload.** The relational grammar (2.4 KB in the
   prototype) is domain-independent and amortises over the whole artifact; the *payload* is
   what scales. We report both numbers, always.
10. **The honest ceiling:** 1 MB cannot hold 1e12 tokens. It can hold ~10⁵–10⁶ relational
    atoms, or — if the domain is derivable — a *program* whose answer space is effectively
    unbounded (measured: 32 atoms → 4,128 tuples → 12,288 queries → 2.2 bits/answer).

---

## SECTION 2: MATHEMATICAL MODEL

### 2.0 Notation

| symbol | meaning |
|---|---|
| `x` | input prompt (token sequence) |
| `e_s` | subject entity (canonical string → id) |
| `r` | relation id |
| `z` | retrieved memory state ("wave state"), ∈ ℂ^{d_w} |
| `θ_DNA` | DNA artifact: (Lattice, Namespace, Lexicon, RuleAlgebra, Grammar) |
| `y` | output tokens |
| `S` | number of cells (slots) in the lattice |
| `D` | carriers per cell |
| `K` | facts per cell (load factor) |
| `p` | payload bytes per atom |
| `R` | replica count (write-sparsification factor) |

### 2.1 Encoding: prompt → (entity, relation) → spiral address

**Parse.** A rule-grammar ofmode` is decided by *specificity ordering*: composed relations (CHAIN) beat inverse
relations beat direct relations; ties broken by match length. This is not cosmetic — with
naive ordering, "capital of the country of Paris" resolves as a broken direct lookup
(measured: chain accuracy 0 % → 100 % after the fix).

**Spiral address.** The addressΦ(x) = (e_s, r, mode)         mode ∈ {DIRECT, INVERSE, CHAIN}Cell lookup walks levels ℓ = 0,1,2,… until the cell whose stored `(entity_id, level)` matches.
This is a **spiral / multi-resolution cell chain**: level 0 holds the first ≤32 atoms of an
entity, level 1 the next, etc. Collisions (two entities hashing to one cell) never overwrite:
the loser spirals to the next level. *Measured effect:* before the collision fix, 6.25 % of
facts were silently destroyed (one country's whole block); after it, 100 % accuracy.

**Why content addressing.** Nothing about `a(e,r)` is stored. The "index" is `H64`. A row
store spends ~30–50 % of its bytes on keys and index entries; HELIX spends 0.

### 2.2 The wave lattice: storage, interference, retrieval

**Cell layout.** A cell is a dense vector of `D` **complex carriers**:A fact `(e, r, o)` is written into cell `c = a(e,r,ℓ)` at carrier index `j = π(r)` where
`π` is a fixed injective map from relation ids to carrier slots (so *relation ids are free* —
they are implied by position). The carrier's **phase** encodes the role code `u_r ∈ {±1, ±i}`
(orthogonal across relations), the **magnitude** encodes the payload `o`.

**Write (superposition with orthogonal role codes).** If several relations of one entity share
a cell, we write them jointly:a(e, r, ℓ) = H64( "cell", id(e), r, ℓ )  mod  S          (ℓ = spiral level)with `u_r` drawn from an orthogonal code family (DFT/Hadamard rows, complex-valued for a
2× phase alphabet). DemodulationC_c ∈ ℂ^D ,      carrier j = ( role_code ⊗ payload ) written as int8 pairsp̂_r = (1/D) · Σ_j conj(u_r[j]) · C_c[j]  =  p_r + Σ_{r'≠r} (⟨u_r,u_{r'}⟩/D) p_{r'}I_r = Σ_{r'≠r} (⟨u_r,u_{r'}⟩/D) p_{r'}     ⇒   Var[I_r] ≈ (K−1) · σ_code² / D² · ‖p‖²
SNR_r = ‖p_r‖ / ‖I_r‖ ≈ √D / (σ_code √(K−1))C_c = Σ_{r ∈ cell(c)} u_r ⊙ p_r ,        p_r = payload vector of relation rso **capacity per cell scales as D²/σ²** in the worst case, and the *usable* load is set by
the ECC, not by the code: if the payload has `t` bits of redundancy, we tolerate
`I` up to the ECC's correction radius. This is the classical **superposition + error-correction
trade** (Kleitman's superimposed codes; coded random access), and it is the only regime in
which superposition is honest.

**Read.**With truly orthogonal `u_r` the cross-term vanishes; with near-orthogonal codes the residual
is the **interference term**, and it is the *only* error source in the system.

**Interference model and SNR.** With `K` atoms in a cell and code length `D`:answer(e, r) :=
if (e, r) ∈ Λ              → payload(e, r)                    [DIRECT, 1 hop]
if ∃ r⁻¹ : r⁻¹(r) = r     → subject such that (·, r, e) ∈ Λ     [INVERSE, 1 hop]
if r = r₁ ∘ r₂ … ∘ r_k     → answer(answer(…(e, r₁), r₂), …, r_k) [CHAIN, k hops]
if r ∈ Gen                → Gen_r(e)                           [GENERATIVE]
if e ∈ alias(e')          → answer(e', r)                      [ALIAS]
else                      → UNKNOWNℓ*` is found by walking the spiral; the walk also yields a *collision confidence*: if the
cell's stored entity fingerprint ≠ `H(id(e))`, the read is rejected (→ "unknown"), which is
how a hash collision is prevented from becoming a *wrong answer*.

**Replica voting (write-sparsification).** Each atom is written to `R` independently-hashed
cells (Kolmogorov–Baramidze). On read, the `R` copies are decoded and majority-voted on
`(relation, object)`; disagreement lowers confidence and can trigger a re-read at the next
spiral level. Interference is independent across replicas, so voting buys ≈ `R/2` dB SNR and,
more importantly, **absence detection**: if all `R` copies are empty, the fact is *absent* with
high confidence — a physical measurement rather than a learned guess.

### 2.3 Rule algebra: 1 atom → K claims

The DNA stores a **rule set** `Γ` (relations, inverses, chains, alias closure, and procedural
generators). Answering is a small term-rewriting evaluation over `Γ`:z = demodulate( C_{a(e,r,ℓ*)} , u_r )            # wave stateEvery derived answer carries the **tuple provenance** `(s, r, o)` of each hop, which is what
lets us compute *semantic* content (Section 4) instead of counting raw queries.

**Generative rules (the 100× regime).** A rule may be a *program* rather than a lookup:
where `idx(e)` is extracted from the entity name by a compiled regex. This is what lets a
3.4 KB artifact answer 12,288 questions: the *namespace itself* is described by a pattern
(`\bp(\d{4})\b` ⇒ `P{:04d}`), so 4,096 entities cost **zero stored bytes** — they are
generated on demand.

### 2.4 Transient parameter generation (the "DNA thinks" step)

The retrieved wave state is mapped to **temporary parameters** that the decoder uses:Gen_r(e) := table[ (idx(e) // q) mod |table| ]        # e.g. idx = trailing digits of eand, in the full version, to a transient low-rank update of the student'sW̃_ℓ = W_ℓ + γ · A_ℓ diag( g_ℓ(z) ) B_ℓᵀ ,      g_ℓ : ℝ^d → ℝ^{r_ℓ}θ_t = A · vec(z) ,        A ∈ ℝ^{d×d} generated from a 32-bit seed  ⇒ 0 stored bytesh_0 = E(x)                                     # encoder (chassis)
h_1 = h_0 + θ_t                                # memory steers the computation
logits = W_out · LN(h_1) + b                   # W_out, b : chassis
y ~ softmax(logits)                           # greedy or sampledFor a knowledge domain the decoder is deliberately tiny: the *hard* part of the answer is
already decided by the memory; the decoder only has to render it. That is why the artifact can
be 12 KB while the teacher is 500 MB: the teacher pays for generality we do not need.

### 2.6 Compilation loss (teacher → DNA)

Compilation is **not** end-to-end
so that **the memory content selects the computation**, rather than merely being concatenated
to it. The student's own weights `W_ℓ` are *chassis* (shared, domain-independent, and
counted separately), while `g_ℓ`, `A_ℓ`, `B_ℓ` are tiny and can live in the artifact.

In the prototype, `θ_t` steers a softmax over answer surface forms, so different memories
produce different phrasings from the same chassis.

### 2.5 DecodingL_compile = λ_kl · KL( p_teacher(· | x) ‖ p_DNA(· | x) )          # behaviour match on probes
+ λ_f  · Σ_{(e,r,o) ∈ facts} 1[ o ≠ teacher(e,r) ]       # factual fidelity
+ λ_gnf· 1[ answer ∈ D ∧ answer ∉ teacher ]             # no hallucination
+ λ_cc  · H(claims(D))                                   # rule induction by MDL
+ λ_spr· ‖ S(D) ‖₁                                      # sparse, non-redundant atoms
+ λ_err· E_cell[ syndrome(cell) ]                       # keep interference in budget1) structural : cell missing, or fingerprint mismatch      → UNKNOWN (hard)
(2) algebraic  : no rule path from (e_s, r) in Γ            → UNKNOWN (hard)
(3) statistical : margin and SNR below calibrated thresholds → UNKNOWN (soft)with the **rule-induction term** `λ_cc` being the important one: the compiler searches for the
smallest rule set `Γ` (measured by description length) such that `Λ ∪ Γ` reproduces the
teacher's answers on a probe set. This is where the *derivation factor* comes from, and it is
what turns a fact list into a knowledge system.

### 2.7 Unknown rejection

Three independent gates, in order of cost:
A calibrated soft gate:`coverage(e_s)` = number of stored atoms for `e_s` (a proxy for "how much do we know about this
entity"), which is exactly the signal a retrieval system needs and which a fine-tuned LLM
never exposes. Measured: 12/12 unknown probes rejected with 0 false accepts, while 100 % of
in-domain queries were answered.

### 2.8 Capacity / knowledge-density model

Let `P` = number of stored scalars in the artifact, `b` = bits per scalarp_unknown = σ( a₁·log SNR_r + a₂·(s₁ − s₂)/‖s‖ + a₃·1[syndrome≠0] + a₄·coverage(e_s) + b )term | meaning | HELIX value (prototype) | LLM value |
|---|---|---|---|
| `b` | bits per stored scalar | 8 (int8 carrier) / 16 (fp16) | 8 / 16 |
| `ρ_c` | **capacity factor** — how many atoms share a scalar | ≈ 1.0 (honest: superposition of *records* does not beat structured storage) | ≈ 1.0 |
| `ρ_q` | **retrieval quality** — fraction of payload that survives interference+ECC | 0.85–0.95 | ≈ 1.0 (no addressing channel to lose) |
| `o_ECC` | ECC + replica overhead | 0.20–0.30 | 0 (weights are "the code") |
| `o_addr` | addressing/index overhead | ≈ 0.0 (content addressing) | folded into the **interpreter tax** |
| `D_derive` | **derivation factor** — answerable claims per stored atom | **1.0 → 129 (measured)** | ≈ 1.0 (a fact is a fact) |
| *interpreter tax* | bits of artifact spent on non-knowledge machinery | ≈ 1.3× | **4× (int8) → 8× (fp16)** |

The interesting line is the last one: **the LLM's `ρ_c·ρ_q` are 1.0, but it pays a 4–8×
tax; HELIX pays 1.3× but can have `D_derive ≫ 1`.** Crossover:Measured: domain A has `D_derive ≈ 1.0` → HELIX *loses* per byte (0.42×). Domain B has
`D_derive = 129` → HELIX wins by 47.9×. **The entire game is `D_derive`, and `D_derive` is a
property of the domain, not of the encoding.** That is the most important sentence in this
document.

---

## SECTION 3: ARCHITECTURE DESIGN

### 3.1 DNA file components (1 MB target)

| component | shape / dtype | bytes @1 MB | role |
|---|---|---|---|
| `HEADER` | magic, version, domain id, section table | 64 | framing |
| `GRAMMAR` | relation patterns + chain/inverse rules (JSON) | 2,400 (fixed) | chassis, amortised |
| `NAMESPACE` | entity id → canonical name (front-coded) | ~30 % of payload | names, once each |
| `LEXICON` | alias/clue → canonical entity; generator tables | ~5 % of payload | surface variation |
| `WAVE_LATTICE` | `S × D` complex int8 cells | ~60 % of payload | the atoms |
| `VALUE_CODEC` | role codes (DFT/Hadamard), RS seeds, replica seeds | ~1 % | decoding params |
| `RS_CODEWORD` | file-level Reed-Solomon over everything above | 6.3 % | corruption control |

Concrete 1 MB configuratiousable_knowledge_bits  ≈  P · b · ρ_c · ρ_q · (1 − o_ECC) · (1 − o_addr) · D_deriveS = 32,768 cells        D = 16 carriers/cell      complex int8 → 1,048,576 B exactly
atoms/cell (load) ≈ 3–8   →  ≈ 100k–260k atoms   (2 replicas → 50k–130k distinct atoms)
bytes/atom ≈ 8–16 B (payload 3 B + fingerprint + ECC + replica overhead)HELIX beats the int8 teacher  ⟺  1.3 · D_derive  >  4     ⟺   D_derive > 3.1