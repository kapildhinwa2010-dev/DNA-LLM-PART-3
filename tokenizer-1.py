"""Byte-level BPE tokenizer (GPT-2 style) matching Qwen2's tokenizer exactly.

Qwen2's pretokenizer regex is
    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|
    \\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
Python's `re` has no \\p{L}/\\p{N}, so we translate them:
    \\p{L} -> [^\\W\\d_]   (letters + marks; Python's \\w minus digits/underscore)
    \\p{N} -> \\d
    "not letter and not number" -> (?:[^\\w]|_)   (underscore is not a letter/number)
The only residual difference from the reference implementation is that Python's
\\w also contains combining marks, which \\p{L} does not — this only shifts
pre-token boundaries for accented text, not for the English probes used here.
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache

PRETOKEN_RE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\w]|_)?[^\W\d_]+"
    r"|\d"
    r"| ?(?:[^\w\s]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

import re
_PRETOKEN = re.compile(PRETOKEN_RE)


@lru_cache(maxsize=1)
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class QwenTokenizer:
    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        merges = [tuple(l.split()) for l in lines[1:] if len(l.split()) == 2]
        self.bpe_ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.decoder = {v: k for k, v in self.encoder.items()}

    # ---------------- BPE core ----------------
    def _bpe(self, token: str) -> list[str]:
        word = tuple(token)
        if len(word) == 1:
            return word
        while True:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            candidate = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if candidate not in self.bpe_ranks:
                break
            first, second = candidate
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return word

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for chunk in _PRETOKEN.findall(text):
            if not chunk:
                continue
            # special tokens are matched literally by the vocab
            if chunk in self.encoder:
                ids.append(self.encoder[chunk])
                continue
            token = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(token):
                ids.append(self.encoder.get(piece, self.encoder.get(self._unk(piece))))
        return ids

    def _unk(self, piece: str) -> str:
        # fall back to the byte-level single chars
        if piece in self.encoder:
            return piece
        return ""

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder.get(i, "") for i in ids)
        raw = bytearray()
        for ch in text:
            if ch in self.byte_decoder:
                raw.append(self.byte_decoder[ch])
        return raw.decode("utf-8", errors="replace")

    # ---------------- chat template (ChatML) ----------------
    def chat_prompt(self, user: str, system: str | None = None) -> str:
        p = ""
        if system:
            p += f"<|im_start|>system\n{system}<|im_end|>\n"
        p += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return p


# Qwen2.5 chat-template special token ids (the repo stores them with empty
# content, so we insert the ids explicitly -- this is what the model was
# trained with).
IM_START = 151644
IM_END = 151645
NEOX = 151643


def chatml_ids(tok: "QwenTokenizer", user: str, system: str | None = None) -> list[int]:
    """Build the ChatML prompt as token ids (specials inserted explicitly)."""
    ids: list[int] = []
    if system:
        ids += [IM_START] + tok.encode("system\n" + system) + [IM_END] + tok.encode("\n")
    ids += [IM_START] + tok.encode("user\n" + user) + [IM_END] + tok.encode("\n")
    ids += [IM_START] + tok.encode("assistant\n")
    return ids
