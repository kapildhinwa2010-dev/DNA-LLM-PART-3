# PRIORITY 1 — the real teacher extraction bridge

**What this closes.** In session 1, `compile_from_teacher(teacher)` took hand-written fact
tables while claiming to compile from a teacher. That was the biggest hole in the project.
This document reports the bridge that actually extracts knowledge from a **real LLM** and
then runs without it.

750 probes  →  Qwen2.5-0.5B-Instruct  →  typed-slot resolution  →  142 atoms
→  KnowledgeDNA (teacher deleted)  →  DNA-only evaluationTeacher: **Qwen2.5-0.5B-Instruct**, run by `nano_qwen/` — a **pure-numpy** inference engine
  (byte-level BPE tokenizer, safetensors reader, Qwen2 forward with GQA/RoPE/SwiGLU, KV
  cache, batched greedy decoding). The sandbox has no torch/transformers and PyPI is
  blocked, so the engine was written from scratch: 988 MB of bf16 weights → a flat fp32
  memmap, left-padded batched decoding, ~2.4 s/probe on 2 CPU cores.
* Probe grid: 30 countries + 30 elements × 5 relations × 5 paraphrases = **750 probes**
  (teacher answered in 1,798 s).
* Evaluation uses a **held-out template set** (different surface forms from the teacher's
  probes) and **the teacher is deleted before the DNA is evaluated**.

---

## 1. Headline numbers

| metric | value |
|---|---|
| teacher exact-answer rate (raw string == gold) | **47.6 %** |
| teacher answer resolves to a valid entity | 63.9 % |
| teacher probe accuracy by relation | capital 87 %, continent 91 %, symbol 67 %, number 59 %, **category 34 %** |
| **extraction precision** | **0.817** |
| **extraction recall** | **0.773** |
| **extraction F1** (vs the probed gold subset) | **0.795** (116/150 tuples) |
| extracted atoms | 142 |
| **DNA-from-teacher accuracy (DNA only, teacher deleted)** | **77.3 %** (232/300 queries) |
| parse failures on the held-out set | **0 / 300** |
| DNA-from-gold accuracy (control) | **100.0 %** |
| **extraction loss** | **22.7 points** |
| artifact size (142 atoms) | 5,919 B (41.7 B/atom)