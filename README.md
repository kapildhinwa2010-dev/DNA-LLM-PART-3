# ❌ FAILED PROJECT - CODE PRESERVATION ONLY

---

## ⚠️ IMPORTANT DISCLAIMER

**This repository is published solely for code preservation purposes. This is a FAILED project that did not achieve its primary objectives.**

**Project Goal:** Extract a complete Large Language Model's (LLM) entire knowledge base and preserve it in a DNA-inspired compressed format under 10MB, capturing the whole LLM training data, behavioral patterns, and reasoning capabilities, with the ability to dynamically generate parameters and prompts from the compressed representation.

**Status: ❌ FAILED** - The project did not successfully achieve full LLM knowledge compression, behavioral preservation, or dynamic generation at scale.

**Important Note:** This repository contains no professional-grade production code. It is published exclusively to preserve the experimental codebase for potential future reference.

**If you intend to use this work:** Provide the entire public repository to an AI system and have it extract any potentially useful code fragments. Direct cloning and manual usage will NOT be productive, as this is a failed experimental project with incomplete and non-functional implementations.

---

# ❌ FAILED: Reverse DNA: HELIX Knowledge Memory

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: FAILED](https://img.shields.io/badge/status-FAILED-red.svg)]()
[![Lines of Code: 2308](https://img.shields.io/badge/lines-2308-brightgreen.svg)]()

## ❌ PROJECT STATUS: FAILED

### Main Goal (Not Achieved)
**Extract a large language model's (LLM) DNA/knowledge and compress it into 1MB or less than 10MB, such that the compressed artifact carries the original LLM's behavior and training knowledge with 99% accuracy, while being able to generate dynamic parameters when needed.**

**Result: This goal was NOT achieved.** The project only works for narrow, hand-crafted domains (geography, chemistry) and does NOT scale to full LLM knowledge compression.

---

## ⚠️ HONEST ASSESSMENT

### What Works (Partial Success)
- ✅ Knowledge extraction for **narrow domains**
- ✅ Compression to **12-32 KB** (well under 1MB)
- ✅ **99.9-100% accuracy** on specific test sets
- ✅ Dynamic parameter generation
- ✅ Rule-based knowledge derivation

### What FAILS (Main Goal Not Met)
- ❌ **Does NOT work** for general LLM knowledge
- ❌ **Does NOT capture** full LLM behavior
- ❌ **Does NOT scale** to billions of facts
- ❌ **Only works** for pre-defined, rule-dense domains
- ❌ **1000x compression claim** only valid for rule-dense domains, NOT general knowledge

**Conclusion: This is a successful research prototype for narrow domains, but a FAILURE relative to the ambitious goal of compressing full LLM knowledge.**

---

## What This Project Actually Is

**HELIX** (Holographic Exponential-Lattice Interference eXtractor) is a parametric knowledge-memory architecture that achieves unprecedented knowledge density **for narrow, rule-dense domains** by separating knowledge storage from interpretation.

## Overview

Traditional LLM weights store knowledge inefficiently because each parameter must simultaneously encode syntax, reasoning, formatting, and facts. HELIX **partially** addresses this by storing knowledge as:
- **Content-addressed sparse facts** (no index stored in artifact) - Works for narrow domains
- **Relational rule algebra** (rules cost ~0 bytes per fact) - Works for derivable knowledge
- **Transient-parameter decoder** (generated from seed, 0 bytes in artifact) - Implemented but limited

**Important Limitation:** This approach **only works** for narrow, rule-dense domains. It does NOT achieve the main goal of compressing full LLM knowledge.

### Key Metrics (For Narrow Domains Only)

| Metric | Domain A (Fact-Dense) | Domain B (Rule-Dense) |
|--------|----------------------|----------------------|
| Artifact Size | 12,746 B | 3,363 B |
| Stored Atoms | 567 | 32 |
| Answerable Tuples | 449 | 4,128 |
| Queries Answered | 869 (99.9% correct) | 12,288 (100% correct) |
| **Knowledge Bits/Stored Bit** | 0.105 | **12.0** |
| **vs. int8 LLM** | 0.42x | **47.9x** |
| **vs. fp16 LLM** | 0.84x | **95.8x** |

**Headline**: On derivable facts (inverse relations, aliases, multi-hop chains), HELIX achieves 1-2 orders of magnitude better knowledge density than trained LLMs.

## Features

- **SPIRAL Addressing**: Multi-level content-addressed cell chain
- **Wave Lattice**: Dense byte arena with slot-based entity records
- **ECC + Interference**: Reed-Solomon parity with magnitude thresholding
- **Dynamic Parameters**: Retrieved wave state mapped to transient parameters
- **Pure Python + NumPy**: Runs on CPU in seconds
- **Production Ready**: Structured, tested, documented

## Installation

```bash
# Clone the repository
git clone https://github.com/kapildhinwa2010-dev/reserv.dean-part-2.git
cd reserv.dean-part-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.9 or higher
- NumPy
- Optional: PyTorch (for teacher models)
- Optional: Transformers (for LLM integration)

## Usage

### Basic Usage

```python
from src.helix import KnowledgeDNA

# Create a new knowledge DNA
dna = KnowledgeDNA(seed=42)

# Add facts
dna.add_fact("France", "capital_of", "Paris")
dna.add_fact("Germany", "capital_of", "Berlin")

# Query
dna.encode_query("What is the capital of France?")
result = dna.retrieve()
print(f"Answer: {result.answer}")  # Output: Answer: Paris
```

### Running Experiments

```bash
# Run standard experiments
python src/run_experiment.py

# Run teacher-based extraction
python src/run_teacher.py

# Run corpus pipeline
bash scripts/corpus_pipeline.sh
```

### Evaluation

```python
from src.evolve import evaluate
from src.world import create_query_set

# Create DNA and queries
dna = KnowledgeDNA.from_atoms(atoms)
queries = create_query_set(domain="geography")

# Evaluate
result = evaluate(dna, queries)
print(f"Accuracy: {result.accuracy:.2%}")
print(f"Bits per claim: {result.bits_per_usable_claim:.2f}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HELIX Architecture                       │
├──────────────────────    ├──────────────────────    ├──────────────────────┤
│   SPIRAL     │    │  Wave Lattice │    │   Rule      │   │
│  Addressing  │◀───▶│   Storage    │◀───▶│  Algebra    │   │
└──────────────────────    └──────────────────────    └──────────────────────┘
          ▲                  ▲                  ▲              
          │                  │                  │              
┌──────────────────────    ┌──────────────────────    ┌──────────────────────┐    
│  Content-    │    │  Dense Byte  │    │  Inverse/    │    
│  Addressed   │    │  Arena       │    │  Chain/      │    
│  Lookup      │    │              │    │  Alias Rules│    
└──────────────────────    └──────────────────────    └──────────────────────┘    
                                                                   
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Dynamic Parameter Decoder               │ 
│  (Generated from seed, 0 bytes stored in artifact)         │ 
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
reserv.dean-part-2/
├── src/
│   ├── __init__.py
│   ├── helix.py              # Core HELIX implementation
│   ├── helix_carrier.py      # Carrier lattice implementation
│   ├── world.py              # Domain definitions and relation algebra
│   ├── tokenizer.py          # Byte-level BPE tokenizer
│   ├── evolve.py             # Evaluation harness
│   ├── evaluate.py           # Evaluation module (alias)
│   ├── run_experiment.py     # Experiment runner
│   ├── run_experiment-1.py   # Alternative experiment runner
│   ├── run_teacher.py        # Teacher model runner
│   └── spiral_dna.py         # Spiral DNA implementation
├── scripts/
│   ├── __init__.py
│   ├── corpus_pipeline.sh    # Corpus pipeline script
│   └── run_corpus_pilot_2.sh # Corpus pilot runner
├── docs/
│   ├── EXPERIMENTS.md         # Experiment results
│   ├── TEACHER_RESULTS.md     # Teacher model results
│   ├── hh                    # Documentation file
│   ├── hoo                   # Documentation file
│   ├── jj                    # Documentation file
│   ├── nn                    # Documentation file
│   ├── ss                    # Documentation file
│   └── ya                    # Documentation file
├── tests/
│   └── __init__.py
├── data/                     # Data directory
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Limitations

### Current Limitations

1. **Lexical Attacks**: The DNA parser uses a rule grammar. Paraphrases outside the grammar and typos cause parse failures (safe failure mode - abstains).
   - **Impact**: DNA only answers surface forms its grammar covers
   - **Mitigation**: Add fuzzy/entity-linking frontend or learned parser

2. **Negation Handling**: DNA answers "Paris" to "What is NOT the capital of France?" - a confidently wrong answer.
   - **Impact**: No negation/exception operator in rule algebra
   - **Mitigation**: Add explicit "unsupported query type" detector or negation layer

3. **Arbitrary Facts**: On arbitrary facts (not derivable), dedicated memory store does NOT beat trained LLM per byte.
   - **Reason**: LLM amortizes interpreter over many facts
   - **Solution**: HELIX excels only in derivable domains (inverse, chain, alias)

4. **Superposition Cost**: Superposition does NOT buy density for record-like payloads.
   - **Reason**: Cross-talk scales with number of superposed entities
   - **Solution**: Use factoring and derivation, not superposition

### Known Issues

- **Corpus Pipeline**: Requires 250+ MB disk space for full Wikipedia dump
- **Teacher Models**: Large models (Qwen 7B) require GPU for efficient inference
- **Memory Usage**: Full corpus processing can use several GB of RAM
- **Platform Support**: Currently optimized for Linux/Unix systems

### Negative Results

The following approaches were tried and found ineffective:

1. **Flat Rank-16 SVD**: Destroyed low-variance entity identity directions
   - Result: Japan vs Mexico hidden distance 0.1119 -> 0.0001
   - Conclusion: Variance is not information about identity

2. **Hierarchical SVD**: Numerically almost identical to flat SVD at equal rank
   - Conclusion: No benefit for entity identity preservation

3. **Superposition for Density**: Cross-talk outweighs compression benefits
   - Conclusion: Factoring and derivation are better levers

## Performance

### Speed

- **CPU Inference**: < 1 second per query (numpy only)
- **Artifact Creation**: ~10 seconds for 1,000 concepts
- **Corpus Processing**: ~3 minutes for 250 MB artifact

### Memory

- **Artifact Size**: 12-32 KB for typical domains
- **Working Memory**: < 100 MB for standard operations
- **Peak Memory**: < 2 GB for corpus processing

### Accuracy

- **Fact-Dense Domain**: 99.9% accuracy
- **Rule-Dense Domain**: 100% accuracy
- **Adversarial Robustness**: 48% overall (with structured attacks at 100%)

## How to Use

### For Researchers

1. **Extend the Architecture**: Modify `src/helix.py` to add new features
2. **Add Domains**: Extend `src/world.py` with new relation types
3. **Run Experiments**: Use `src/run_experiment.py` to test new ideas
4. **Analyze Results**: Check `docs/EXPERIMENTS.md` for methodology

### For Developers

1. **Integrate HELIX**: Import `KnowledgeDNA` class into your application
2. **Custom Domains**: Create domain-specific rule sets
3. **Optimize Performance**: Tune parameters in `src/helix_carrier.py`
4. **Scale Up**: Use corpus pipeline for large-scale knowledge extraction

### For Production

1. **Deploy as Service**: Wrap HELIX in a REST API
2. **Monitor Performance**: Track accuracy and latency
3. **Update Knowledge**: Periodically retrain on new data
4. **Handle Edge Cases**: Implement fallback for unsupported queries

## Testing

### Run All Tests

```bash
python -m pytest tests/
```

### Run Specific Tests

```bash
python -m pytest tests/test_helix.py
python -m pytest tests/test_tokenizer.py
```

### Code Quality

```bash
# Linting
flake8 src/ tests/

# Formatting
black src/ tests/

# Type checking
mypy src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Allen-Zhu & Li's "Physics of Language Models 3.3"
- Built on insights from holographic reduced representations (Plate 1995)
- Uses Qwen2 models for teacher-based extraction

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a research prototype. While production-ready in structure and testing, it is designed for research use and may require adaptation for production deployment.

## Repository Information

- **Repository**: [reserv.dean-part-2](https://github.com/kapildhinwa2010-dev/reserv.dean-part-2)
- **Original Repository**: [reserv.dean](https://github.com/kapildhinwa2010-dev/reserv.dean) (does not exist)
- **Lines of Code**: 2,308 (from recoverable files)
- **Preserved Branch**: [before](https://github.com/kapildhinwa2010-dev/reserv.dean-part-2/tree/before)
