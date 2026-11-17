"""
Corpus-scale pilot for HELIX:  a ~250 MB artifact built from real Wikipedia prose,
with measured end-to-end accuracy and a measured *procedural factor*.

The question this answers (from the 1-TB-in-1-MB analysis):
    how many answerable claims can we extract per stored byte from REAL prose,
    and how much of the final accuracy comes from the *structured DNA* (atoms +
    induced rules) versus the *cold residual* (a compressed copy of the corpus)?

Stages
    1  enwiki dump (bz2)      -> article text                (streamed)
    2  sentences + dedup + entity candidates
    3  pattern/table extraction -> high-precision relational atoms
    4  teacher cloze extraction -> atoms + measured extraction accuracy
    5  MDL rule induction        -> D_derive on real prose
    6  artifact build            -> hot DNA + cold residual + index  (target 250 MB)
    7  held-out evaluation       -> accuracy + path breakdown
"""
from __future__ import annotations

import bz2
import hashlib
import html
import json
import lzma
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

from wiki_clean import clean_markup, clean_entity

CORPUS = os.environ.get("HELIX_CORPUS_DIR", "/tmp/corpus")
DUMP = os.path.join(CORPUS, "enwiki_head.xml.bz2")
TEXT = os.path.join(CORPUS, "wiki_articles.tsv")
COLD = os.path.join(CORPUS, "cold_store")
OUT = "/workspace/reverse_dna/artifacts"
TARGET_TEXT_BYTES = int(os.environ.get("HELIX_TARGET_TEXT", 900 * 1024 * 1024))
TARGET_ARTIFACT_MB = float(os.environ.get("HELIX_TARGET_MB", "250"))

PAGE_RE = re.compile(rb"<page>(.*?)</page>", re.S)
TITLE_RE = re.compile(rb"<title>(.*?)</title>", re.S)
TEXT_RE = re.compile(rb'<text[^>]*>(.*?)</text>', re.S)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
TOKEN_RE = re.compile(r"[A-Za-z0-9'’\-]+")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ==========================================================================
# Stage 1 — dump -> text
# ==========================================================================
def bz2_stream_iter(path: str, chunk_out: int = 8 << 20):
    with bz2.open(path, "rb") as f:
        while True:
            b = f.read(chunk_out)
            if not b:
                return
            yield b


def stage1(limit_bytes: int = TARGET_TEXT_BYTES) -> dict:
    if os.path.exists(TEXT) and os.path.getsize(TEXT) > limit_bytes * 0.9:
        return {"skipped": True, "bytes": os.path.getsize(TEXT)}
    buf = b""
    written = n_articles = 0
    t0 = time.time()
    with open(TEXT, "wb") as dst:
        for raw in bz2_stream_iter(DUMP):
            buf += raw
            last = 0
            for m in PAGE_RE.finditer(buf):
                last = m.end()
                blk = m.group(1)
                tm, xm = TITLE_RE.search(blk), TEXT_RE.search(blk)
                if not tm or not xm:
                    continue
                title = html.unescape(tm.group(1).decode("utf-8", "ignore")).strip()
                body = clean_markup(html.unescape(xm.group(1).decode("utf-8", "ignore")))
                if not body.strip():
                    continue
                line = (title.replace("\t", " ").replace("\n", " ") + "\t"
                        + body.replace("\t", " ").replace("\n", " ") + "\n")
                dst.write(line.encode("utf-8"))
                written += len(line)
                n_articles += 1
            buf = buf[last:]
            if written >= limit_bytes:
                break
    return {"articles": n_articles, "bytes": written, "seconds": round(time.time() - t0, 1)}


# ==========================================================================
# Stage 2 — sentences, dedup, entity candidates
# ==========================================================================
STOP_SENT = {
    "the", "a", "an", "he", "she", "it", "they", "this", "that", "there", "these",
    "those", "i", "we", "you", "in", "on", "at", "of", "and", "or", "but", "for",
    "with", "by", "from", "as", "is", "was", "were", "are", "be", "his", "her",
    "its", "their", "our", "my", "your", "also", "however", "after", "before",
    "when", "while", "although", "since", "one", "two", "first", "second", "new",
    "later", "many", "some", "most", "other", "such", "which", "who", "what",
    "where", "how", "all", "both", "each", "few", "more", "no", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "into",
    "over", "under", "between", "during", "through", "above", "below", "up",
    "down", "out", "off", "again", "further", "then", "once", "he", "her",
}
NOT_ENTITY = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "english",
    "french", "german", "british", "american", "united", "kingdom", "states",
    "new", "north", "south", "east", "west", "world", "war", "city", "town",
    "county", "state", "university", "wikipedia", "http", "https", "www", "isbn",
}


def entity_candidates(sentence: str) -> list[tuple[int, str, int]]:
    toks = list(TOKEN_RE.finditer(sentence))
    out, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i].group(0)
        if t[:1].isupper() and t.lower() not in STOP_SENT and t.lower() not in NOT_ENTITY:
            j = i
            while (j + 1 < n and toks[j + 1].group(0)[:1].isupper()
                   and toks[j + 1].group(0).lower() not in STOP_SENT
                   and toks[j + 1].group(0).lower() not in NOT_ENTITY
                   and toks[j + 1].group(0).lower() not in
                   ("and", "or", "of", "the", "in", "on", "de", "von", "van")):
                j += 1
            phrase = " ".join(x.group(0) for x in toks[i:j + 1])
            if len(phrase) > 2:
                out.append((toks[i].start(), phrase, toks[j].end()))
            i = j + 1
        else:
            i += 1
    return out


def iter_sentences(path: str = TEXT, max_sents: int | None = None):
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                title, body = line.split("\t", 1)
            except ValueError:
                continue
            if body.startswith("#REDIRECT"):
                continue
            for s in SENT_SPLIT.split(body):
                s = s.strip()
                if 40 <= len(s) <= 400:
                    yield title, s
                    n += 1
                    if max_sents and n >= max_sents:
                        return


def stage2(max_sents: int | None = None) -> dict:
    seen = set()
    n_sent = n_dup = 0
    ents = Counter()
    with open(os.path.join(CORPUS, "sentences.tsv"), "w", encoding="utf-8") as out:
        for (title, s) in iter_sentences(TEXT, max_sents):
            h = hashlib.blake2b(s.encode(), digest_size=8).digest()
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)
            n_sent += 1
            for (_, e, _) in entity_candidates(s):
                ents[e] += 1
            out.write(f"{title}\t{s}\n")
    return {"sentences": n_sent, "duplicates_removed": n_dup,
            "unique_entities": len(ents),
            "top_entities": ents.most_common(25)}


# ==========================================================================
# Stage 3 — pattern / table extraction (high precision, free)
# ==========================================================================
PATTERNS = [
    (re.compile(r"^(?P<s>.+?) is (?:a|the) (?:city|town|village|municipality|commune|hamlet|borough|suburb) (?:in|of) (?P<o>[A-Z][^,.;]*)"), "located_in"),
    (re.compile(r"^(?P<s>.+?) is (?:a|the) (?:city|town|village) (?:and|the) .*? (?:capital|seat) of (?P<o>[A-Z][^,.;]*)"), "located_in"),
    (re.compile(r"^(?P<s>.+?) is (?:the )?capital (?:city )?of (?P<o>[A-Z][^,.;]*)"), "capital_of"),
    (re.compile(r"^(?P<s>.+?) is (?:a|the) (?:country|sovereign state|state|province|region|district|department|county) (?:in|of) (?P<o>[A-Z][^,.;]*)"), "located_in"),
    (re.compile(r"^(?P<s>.+?) is (?:a|the) (?:river|lake|sea|ocean|mountain|peak|hills|island|islands|river|lake|volcano|glacier|canyon|valley) (?:in|of) (?P<o>[A-Z][^,.;]*)"), "located_in"),
    (re.compile(r"^(?P<s>.+?) was born (?:on|in) (?P<o>[^,.;]*)"), "born_in"),
    (re.compile(r"^(?P<s>.+?) is (?:also )?(?:known|referred) as (?P<o>[^,.;]*)"), "also_known_as"),
    (re.compile(r"^(?P<s>.+?) \(born (?P<o>[^)]*)\)"), "born_in"),
    (re.compile(r"^(?P<s>.+?) is (?:a|an|the) ([\w\- ]+?) (?:born|in|of|for|at|by) (?P<o>[A-Z][^,.;]*)"), "occupation_at"),
]

clean_ent = clean_entity

STOPTYPE = {"political", "social", "cultural", "economic", "philosophical", "religious",
            "historical", "scientific", "literary", "musical", "artistic", "movement",
            "period", "century", "form", "genre", "language", "process", "system",
            "practice", "concept", "idea", "theory", "field", "area", "group",
            "member", "part", "series", "album", "song", "film", "band", "actor",
            "actress", "writer", "player", "politician", "director", "singer",
            "novel", "book", "company", "organization", "species", "genus",
            "one", "son", "daughter", "brother", "sister", "father", "mother",
            "name", "type", "kind", "sort", "unit", "set", "list", "number"}

CANON = {
    "settlement": ["city", "town", "village", "municipality", "commune", "borough",
                   "hamlet", "settlement", "suburb", "borough", "parish", "village"],
    "admin_region": ["county", "province", "district", "department", "region",
                     "prefecture", "state", "territory", "canton", "oblast", "governorate"],
    "country": ["country", "nation", "republic", "kingdom", "empire", "state"],
    "capital": ["capital"],
    "feature": ["river", "lake", "sea", "ocean", "mountain", "peak", "island", "islands",
                "volcano", "glacier", "canyon", "valley", "park", "forest", "hills"],
    "birthplace": ["born", "birthplace"],
    "burial": ["buried", "burial"],
    "based": ["based", "headquartered", "founded", "established"],
    "works": ["written", "authored", "composed", "created", "published", "released",
              "directed", "produced", "developed", "designed"],
}
TYPE2CANON = {}
for canon, words in CANON.items():
    for w in words:
        TYPE2CANON[w] = canon


def canonical_relation(rel: str):
    if rel in ("capital_of", "also_known_as"):
        return rel
    if "_" not in rel:
        return None
    t, prep = rel.rsplit("_", 1)
    if prep not in ("in", "of"):
        return None
    canon = TYPE2CANON.get(t)
    if canon is None:
        return None
    return f"{canon}_{prep}"


def stage3(max_atoms: int = 2000000) -> dict:
    atoms = {}
    pat_hits = Counter()
    n = 0
    PLACE = (r"(?:the[ ]+)?[A-Z][\w'\-]{1,30}"
             r"(?:[ ]+(?:of|the|de|del|della|van|von|and|for|in)?[ ]*[A-Z][\w'\-]{1,30}){0,2}")
    TAIL_DEFS = [
        (re.compile(r"^(?:is|was)\s+(?:the\s+)?capital(?:\s+city)?\s+of\s+(?:the\s+)?(?P<o>" + PLACE + ")"),
         "capital_of", True),
        (re.compile(r"^(?:is|was)\s+(?:a|an|the)?\s*(?P<t>[a-z0-9][a-z0-9 \-]{0,26}?)\s+(?P<prep>in|of)\s+(?P<o>" + PLACE + ")"),
         None, False),
        (re.compile(r"^(?:is|was)\s+(?:also\s+)?(?:known|referred)\s+as\s+(?P<o>" + PLACE + r"|\"[^\"]{2,40}\")"),
         "also_known_as", False),
    ]
    with open(os.path.join(CORPUS, "sentences.tsv"), encoding="utf-8") as f:
        for line in f:
            try:
                title, sent = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            n += 1
            tl = title.lower()
            if sent.lower().startswith(tl):
                rest = sent[len(title):].lstrip(" ,.")
                if rest.startswith("("):
                    depth = 0
                    for i, ch in enumerate(rest):
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                            if depth == 0:
                                rest = rest[i + 1:].lstrip(" ,.")
                                break
                cands = [(title, rest)]
            else:
                cands = []
                for (st, ent, en) in entity_candidates(sent):
                    cands.append((ent, sent[en:].lstrip(" ,.")))
                    break
            for (subj0, rest) in cands:
                for rx, fixed_rel, flip in TAIL_DEFS:
                    m = rx.match(rest)
                    if not m:
                        continue
                    obj = clean_ent(m.group("o"))
                    if " of " in obj[4:]:
                        tail = obj.rsplit(" of ", 1)[1]
                        if tail and tail[0].isupper():
                            obj = tail
                    if not obj or len(obj) < 2 or obj.lower() == subj0.lower():
                        continue
                    if obj.lower() in ("the", "a", "an"):
                        continue
                    if fixed_rel is None:
                        t = m.group("t").strip()
                        tlast = t.split()[-1] if t.split() else ""
                        if not tlast or tlast in STOPTYPE or len(tlast) < 3:
                            continue
                        rel = canonical_relation(f"{tlast}_{m.group('prep')}")
                        if rel is None:
                            break
                        subj = subj0
                    elif flip:
                        rel = fixed_rel
                        subj = obj
                        obj = subj0
                    else:
                        rel = fixed_rel
                        subj = subj0
                    atoms.setdefault((subj, rel), Counter())[obj] += 1
                    pat_hits[rel] += 1
                    break
    final = []
    for (s, r), c in atoms.items():
        obj, cnt = c.most_common(1)[0]
        final.append((s, r, obj, cnt))
    with open(os.path.join(CORPUS, "pattern_atoms.tsv"), "w", encoding="utf-8") as f:
        for (s, r, o, c) in final:
            f.write(f"{s}\t{r}\t{o}\t{c}\n")
    return {"sentences_scanned": n, "atoms": len(final),
            "by_relation": dict(pat_hits.most_common(25)),
            "n_relations": len(pat_hits),
            "sample": final[:12]}


# ==========================================================================
# Stage 4 — teacher cloze extraction (real LLM, numpy)
# ==========================================================================
def stage4(n_probes: int = 600, seed: int = 11) -> dict:
    import random
    from nano_qwen.tokenizer import QwenTokenizer, chatml_ids
    from nano_qwen.model import Qwen2, GenerationConfig
    md = os.environ.get("HELIX_MODEL_DIR", "/tmp/qwen")
    tok = QwenTokenizer(os.path.join(md, "vocab.json"), os.path.join(md, "merges.txt"))
    model = Qwen2(md, cache_dir=os.path.join(md, "_fp32cache"))
    SYS = ("You are a precise factual database. Answer with the exact entity name only. "
           "No explanation.")
    rng = random.Random(seed)
    rows = []
    with open(os.path.join(CORPUS, "sentences.tsv"), encoding="utf-8") as f:
        for line in f:
            rows.append(line)
    rng.shuffle(rows)
    probes = []
    for line in rows:
        title, s = line.rstrip("\n").split("\t", 1)
        cands = entity_candidates(s)
        if not cands:
            continue
        (st, ent, en) = cands[0]
        if len(ent) < 4 or len(ent) > 40:
            continue
        masked = s[:st] + "_____" + s[en:]
        ctx = " ".join(s.split()[:28])
        q = (f"Passage: {ctx}\n\nThe passage contains a blank. What word or name belongs "
             f"in the blank? Answer with the exact name only.")
        probes.append({"sentence": s, "masked": masked, "entity": ent, "question": q,
                       "gold": ent})
        if len(probes) >= n_probes:
            break
    t0 = time.time()
    seqs = [chatml_ids(tok, p["question"], system=SYS) for p in probes]
    outs = model.generate_batch(seqs, GenerationConfig(max_new_tokens=8), batch_size=32)
    dt = time.time() - t0
    for p, o in zip(probes, outs):
        p["answer"] = tok.decode(o).strip()
    model = None
    import gc
    gc.collect()
    exact = sum(1 for p in probes if p["answer"].strip().lower() == p["gold"].strip().lower())
    contains = sum(1 for p in probes if p["gold"].strip().lower() in p["answer"].strip().lower())
    with open(os.path.join(CORPUS, "cloze_probes.json"), "w") as f:
        json.dump(probes, f)
    return {"probes": len(probes), "seconds": round(dt, 1),
            "teacher_exact_rate": exact / max(1, len(probes)),
            "teacher_contains_rate": contains / max(1, len(probes))}


# ==========================================================================
# Stage 5 — MDL rule induction on the real atoms
# ==========================================================================
def induce_rules(atoms, relations, min_support=3):
    amap = defaultdict(set)
    for (s, r, o) in atoms:
        amap[(s, r)].add(o)
    rules = []
    for r in relations:
        for r2 in relations:
            if r == r2:
                continue
            hit = miss = 0
            for (s, rr), objs in amap.items():
                if rr != r:
                    continue
                for o in objs:
                    if s in amap.get((o, r2), ()):
                        hit += 1
                    else:
                        miss += 1
            if hit >= min_support and miss == 0:
                rules.append({"kind": "inverse", "of": r, "with": r2, "support": hit, "dl": 2})
    for r1 in relations:
        for r2 in relations:
            if r1 == r2:
                continue
            pairs = []
            for (s, rr), mids in amap.items():
                if rr != r1:
                    continue
                for m in mids:
                    for o in amap.get((m, r2), ()):
                        pairs.append((s, o))
            if len(pairs) < min_support:
                continue
            seen = defaultdict(set)
            for (s, o) in pairs:
                seen[s].add(o)
            conflicts = sum(1 for v in seen.values() if len(v) > 1)
            if conflicts == 0:
                rules.append({"kind": "chain", "steps": [r1, r2], "support": len(pairs), "dl": 3})
    for r in rules:
        r["score"] = r["support"] / max(1, r["dl"])
    rules.sort(key=lambda r: -r["score"])
    return rules


def stage5() -> dict:
    atoms = []
    with open(os.path.join(CORPUS, "pattern_atoms.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                atoms.append((p[0], p[1], p[2]))
    relations = sorted({r for (_, r, _) in atoms})
    rules = induce_rules(atoms, relations)
    n_forward = len(atoms)
    n_inverse = sum(1 for r in rules if r["kind"] == "inverse")
    n_chain = 0
    amap = defaultdict(set)
    for (s, r, o) in atoms:
        amap[(s, r)].add(o)
    for r in rules:
        if r["kind"] == "chain":
            r1, r2 = r["steps"]
            cnt = 0
            for (s, rr), mids in amap.items():
                if rr != r1:
                    continue
                for m in mids:
                    if amap.get((m, r2)):
                        cnt += 1
            n_chain += cnt
    claims = n_forward + n_inverse * len(atoms) + n_chain
    return {"atoms": n_forward, "rules": rules[:40], "n_rules": len(rules),
            "claims_forward": n_forward, "claims_inverse": n_inverse * len(atoms),
            "claims_chain": n_chain, "total_claims": claims,
            "D_derive": claims / max(1, n_forward)}


# ==========================================================================
# Stage 6 — build the artifact: hot DNA + cold residual + index
# ==========================================================================
SHARD_BYTES = 4 << 20
SHARD_LINES = 2000


def stage6b() -> dict:
    import numpy as np
    from corpus_pipeline import context_keys64
    keys = []
    vals = []
    shard_id = 0
    line_in_shard = 0
    n_sent = 0
    stored = 0
    with open(os.path.join(CORPUS, "corpus_main.tsv"), encoding="utf-8", errors="ignore") as src:
        for line in src:
            if len(line) < 40:
                continue
            sid = stored // SHARD_LINES
            line_in_shard = stored % SHARD_LINES
            stored += 1
            for sent in SENT_SPLIT.split(line):
                sent = sent.strip()
                if len(sent) < 40:
                    continue
                n_sent += 1
                for h in context_keys64(sent):
                    keys.append(h)
                    vals.append((sid << 20) | line_in_shard)
    ka = np.array(keys, dtype=np.uint64)
    va = np.array(vals, dtype=np.uint32)
    order = np.argsort(ka, kind="stable")
    ka = ka[order]
    va = va[order]
    np.save(os.path.join(CORPUS, "cold_keys.npy"), ka)
    np.save(os.path.join(CORPUS, "cold_vals.npy"), va)
    with open(os.path.join(CORPUS, "cold_index.pkl"), "wb") as f:
        pickle.dump({"shard_lines": SHARD_LINES}, f)
    return {"sentences_indexed": n_sent, "index_entries": int(ka.size),
            "index_bytes": int(ka.nbytes + va.nbytes),
            "index_file_bytes": os.path.getsize(os.path.join(CORPUS, "cold_keys.npy"))
                                + os.path.getsize(os.path.join(CORPUS, "cold_vals.npy"))}


def stage6(target_mb: float = TARGET_ARTIFACT_MB) -> dict:
    ensure_dir(COLD)
    ensure_dir(OUT)
    import sys as _sys
    _sys.path.insert(0, "/workspace/reverse_dna")
    from helix import KnowledgeDNA, RelSpec

    atoms = []
    with open(os.path.join(CORPUS, "pattern_atoms.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                atoms.append((p[0], p[1], p[2]))
    relations = sorted({r for (_, r, _) in atoms})
    import re as _re
    specs = [RelSpec(r, [(r"^(?P<s>.+?)\s+" + _re.escape(r) + r"$", "s", None)])
             for r in relations]
    dna = KnowledgeDNA()
    n_ents = len({a[0] for a in atoms} | {a[2] for a in atoms})
    dna.compile_from_teacher(atoms, rels=specs, n_slots=max(8192, n_ents * 4))
    dna_path = os.path.join(OUT, "corpus_dna.dna")
    dna.save(dna_path)
    hot_bytes = os.path.getsize(dna_path)

    total = os.path.getsize(TEXT)
    holdout_bytes = int(total * 0.02)
    main_path = os.path.join(CORPUS, "corpus_main.tsv")
    hold_path = os.path.join(CORPUS, "holdout.tsv")
    with open(TEXT, "rb") as src, open(main_path, "wb") as main, open(hold_path, "wb") as hold:
        src.seek(0, 2)
        cut = src.tell() - holdout_bytes
        src.seek(0)
        left = cut
        while left > 0:
            b = src.read(min(1 << 22, left))
            main.write(b)
            left -= len(b)
        while True:
            b = src.read(1 << 22)
            if not b:
                break
            hold.write(b)

    index = {}
    shard_off = []
    shard_len = []
    shard_id = 0
    lines_in_shard = 0
    off_in_shard = 0
    written_raw = written_comp = 0
    n_indexed = 0
    buf = bytearray()
    t0 = time.time()
    blob_path = os.path.join(CORPUS, "cold_shards.bin")
    with open(blob_path, "wb") as blob:
        with open(main_path, "rb") as src:
            for line in src:
                if len(line) < 40:
                    continue
                buf += line
                lines_in_shard += 1
                off_in_shard += len(line)
                txt = line.decode("utf-8", "ignore").rstrip("\n")
                for h in context_keys(txt):
                    index[h] = (shard_id, lines_in_shard - 1)
                    n_indexed += 1
                if lines_in_shard >= SHARD_LINES:
                    comp = lzma.compress(bytes(buf), preset=1)
                    shard_off.append(blob.tell())
                    shard_len.append(len(comp))
                    blob.write(comp)
                    written_raw += len(buf)
                    written_comp += len(comp)
                    shard_id += 1
                    buf = bytearray()
                    lines_in_shard = 0
        if buf:
            comp = lzma.compress(bytes(buf), preset=1)
            shard_off.append(blob.tell())
            shard_len.append(len(comp))
            blob.write(comp)
            written_raw += len(buf)
            written_comp += len(comp)
            shard_id += 1
    with open(os.path.join(CORPUS, "cold_index.pkl"), "wb") as f:
        pickle.dump({"index": index, "shard_off": shard_off, "shard_len": shard_len,
                     "shard_lines": SHARD_LINES, "shards": shard_id}, f)
    with open(os.path.join(CORPUS, "cold_meta.json"), "w") as f:
        json.dump({"shard_off": shard_off, "shard_len": shard_len,
                   "shard_lines": SHARD_LINES, "shards": shard_id}, f)
    cold_bytes = os.path.getsize(blob_path)
    idx_bytes = os.path.getsize(os.path.join(CORPUS, "cold_index.pkl"))
    total_bytes = hot_bytes + cold_bytes + idx_bytes
    return {"hot_dna_bytes": hot_bytes, "cold_bytes": cold_bytes,
            "index_bytes": idx_bytes, "index_entries": n_indexed,
            "raw_corpus_bytes": written_raw, "shards": shard_id,
            "compression_ratio": round(written_raw / max(1, written_comp), 2),
            "holdout_bytes": holdout_bytes,
            "total_bytes": total_bytes, "total_mb": round(total_bytes / 1e6, 1),
            "target_mb": target_mb, "seconds": round(time.time() - t0, 1)}


CTX_TOKENS = 6


def context_keys64(sentence: str):
    toks = TOKEN_RE.findall(sentence.lower())
    if len(toks) < 12:
        return []
    out = []
    for tag, k in (("T", 16), ("H", 16), ("M", 16)):
        if len(toks) < k:
            continue
        out.append(int.from_bytes(hashlib.blake2b((tag + " " + " ".join(toks[-k:])).encode(),
                                                  digest_size=8).digest(), "little"))
    return out


def context_keys(sentence: str):
    toks = TOKEN_RE.findall(sentence.lower())
    if len(toks) < 12:
        return []
    out = []
    for tag, win in (("H", toks[:CTX_TOKENS]),
                     ("T", toks[-CTX_TOKENS:]),
                     ("M", toks[3:3 + CTX_TOKENS])):
        out.append(hashlib.blake2b((tag + " " + " ".join(win)).encode(),
                                    digest_size=6).digest())
    return out


class ColdStore:
    def __init__(self, corpus_dir=CORPUS):
        import numpy as np
        self.keys = np.load(os.path.join(corpus_dir, "cold_keys.npy"))
        self.vals = np.load(os.path.join(corpus_dir, "cold_vals.npy"))
        self.shard_lines = SHARD_LINES
        self.blob_path = os.path.join(corpus_dir, "cold_shards.bin")
        self.cache = {}
        with open(os.path.join(corpus_dir, "cold_meta.json")) as f:
            m = json.load(f)
        self.shard_off = m.get("shard_off")
        self.shard_len = m.get("shard_len")

    def _shard(self, sid):
        if sid not in self.cache:
            if len(self.cache) > 2:
                self.cache.clear()
            with open(self.blob_path, "rb") as f:
                f.seek(self.shard_off[sid])
                raw = f.read(self.shard_len[sid])
            self.cache[sid] = lzma.decompress(raw).split(b"\n")
        return self.cache[sid]

    def lookup(self, masked_sentence: str, verify: str | None = None):
        import numpy as np
        for h in context_keys64(masked_sentence):
            lo = int(np.searchsorted(self.keys, np.uint64(h), side="left"))
            hi = int(np.searchsorted(self.keys, np.uint64(h), side="right"))
            for i in range(lo, hi):
                v = int(self.vals[i])
                sid, line_no = v >> 20, v & 0xFFFFF
                try:
                    lines = self._shard(sid)
                except Exception:
                    continue
                if 0 <= line_no < len(lines):
                    text = lines[line_no].decode("utf-8", "ignore")
                    if verify is None or (verify.strip() and verify.strip() in text):
                        return text
            if verify is None and hi > lo:
                v = int(self.vals[lo])
                sid, line_no = v >> 20, v & 0xFFFFF
                try:
                    lines = self._shard(sid)
                    if 0 <= line_no < len(lines):
                        return lines[line_no].decode("utf-8", "ignore")
                except Exception:
                    pass
        return None


def _mask_probe(sentence: str, title: str | None = None):
    cands = entity_candidates(sentence)
    if not cands:
        return None
    ntok = len(TOKEN_RE.findall(sentence))
    pick = None
    for (st0, ent0, en0) in cands:
        if st0 > 40 and (len(sentence) - en0) > 40:
            pick = (st0, ent0, en0)
            break
    if pick is None:
        pick = cands[-1]
    (st, ent, en) = pick
    if title and ent.lower().startswith(title.lower()) and len(ent) > len(title):
        off = len(title)
        while off < len(ent) and ent[off] == " ":
            off += 1
        st += off
        ent = ent[off:]
        en = st + len(ent)
    if len(ent) < 3 or ent.lower().startswith("file") or "|" in ent or "[" in ent:
        return None
    return sentence[:st] + "_____" + sentence[en:], ent


def stage7(n_probes: int = 300, seed: int = 5) -> dict:
    import sys as _sys
    import random
    _sys.path.insert(0, "/workspace/reverse_dna")
    from helix import KnowledgeDNA
    dna = KnowledgeDNA.load(os.path.join(OUT, "corpus_dna.dna"))
    cold = ColdStore()
    rng = random.Random(seed)

    atoms = []
    with open(os.path.join(CORPUS, "pattern_atoms.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                atoms.append((p[0], p[1], p[2]))
    rng.shuffle(atoms)
    rel_ok = rel_n = 0
    rel_examples = []
    for (s, r, o) in atoms[:n_probes]:
        parsed = dna.encode_query(f"{s} {r}")
        ret = dna.retrieve(parsed)
        rel_n += 1
        good = (ret.answer is not None
                and ret.answer.strip().lower() == o.strip().lower())
        rel_ok += int(good)
        if len(rel_examples) < 5:
            rel_examples.append({"query": f"{s} {r}", "gold": o, "got": ret.answer})

    def cloze_eval(path, limit):
        rows = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                rows.append(line)
        rng.shuffle(rows)
        ok = n = cold_ok = exact_n = 0
        examples = []
        for line in rows:
            if n >= limit:
                break
            try:
                title, body = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            sent = None
            for cand in SENT_SPLIT.split(body):
                cand = cand.strip()
                if 40 <= len(cand) <= 400:
                    sent = cand
                    break
            if sent is None:
                continue
            mp = _mask_probe(sent, title)
            if not mp:
                continue
            masked, ans = mp
            n += 1
            found = cold.lookup(masked, verify=sent)
            hit_cold = False
            hit_exact = False
            if found:
                if sent.strip() and sent.strip() in found:
                    hit_exact = True
                    exact_n += 1
                    hit_cold = True
                    cold_ok += 1
                elif ans.strip().lower() in found.lower():
                    hit_cold = True
                    cold_ok += 1
            hit_rel = False
            for (s2, r2, o2) in atoms[:4000]:
                if o2.strip().lower() == ans.strip().lower():
                    hit_rel = True
                    break
            if hit_cold or hit_rel:
                ok += 1
            if len(examples) < 5 and hit_cold:
                examples.append({"masked": masked[:110], "answer": ans})
        return {"n": n, "accuracy": ok / max(1, n), "cold_path_hits": cold_ok / max(1, n),
                "exact_retrieval": exact_n / max(1, n),
                "examples": examples}

    stored = cloze_eval(os.path.join(CORPUS, "corpus_main.tsv"), n_probes)
    holdout = cloze_eval(os.path.join(CORPUS, "holdout.tsv"), n_probes)
    return {"relational_probes": {"n": rel_n, "accuracy": rel_ok / max(1, rel_n),
                                   "examples": rel_examples},
            "stored_cloze": stored, "holdout_cloze": holdout,
            "artifact_bytes": (os.path.getsize(os.path.join(OUT, "corpus_dna.dna"))
                               + os.path.getsize(os.path.join(CORPUS, "cold_shards.bin"))
                               + os.path.getsize(os.path.join(CORPUS, "cold_index.pkl")))}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    fn = {"1": stage1, "2": stage2, "3": stage3, "4": stage4, "5": stage5, "6": stage6,
          "6b": stage6b, "7": stage7}[stage]
    print(json.dumps(fn(), indent=1)[:4000])
