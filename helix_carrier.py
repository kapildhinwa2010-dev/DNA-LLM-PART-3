"""
CarrierLattice — the *real* wave lattice (and an honest measurement of what
superposition actually costs).

Design
------
A cell is D complex carriers stored as int16 (real, imag).  An entity owns a
cell (hash-addressed; collisions spiral to the next level).  A fact
(e, r, o) is written at carrier j = r mod D as

        C[j] += u_e[j] * p(o)

where u_e is a deterministic unit-modulus *phase code* for the entity (seeded
by its id -> 0 bytes stored) and p(o) is the 16-bit object reference packed
into the complex plane.

Reading demodulates with the conjugate code (a matched filter):

        p_hat = (1/D) * sum_j conj(u_e[j]) * C[j]

The signal is recovered at scale (matched-filter gain), while *noise* injected
into the carriers is suppressed by sqrt(D) relative to the signal.  The
"absence" of a relation is a physical measurement: if no fact was written at
carrier j, the demodulated value there is noise, and the SNR of the
demodulated vector (signal vs. median noise magnitude) tells us whether the
fact exists.  This is what makes "I don't know" a measurement instead of a
classifier guess.

Superposition mode (optional, and measured in the experiments)
-------------------------------------------------------------
If several entities share a cell, their payloads superpose in the same
carriers.  Writing entity e' adds u_e'[j] p', so the demodulated value for e
becomes

        p_hat = p + (1/D) sum_k conj(u_e[k]) u_e'[k] p'   ->   interference
                    ~ p / sqrt(D)   (random phases)

i.e. the classic holographic cross-talk.  We keep the mode switchable because
the *cost* of superposition is the interesting measurement: to superpose
L entities in one cell you need D carriers per entity (the code dimension),
so the storage per fact grows by ~D complex numbers.  In other words:
superposition buys interference, not density.  `experiments.py` measures both.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np


def entity_code(entity_id: int, D: int, seed: int = 0xC0FFEE) -> np.ndarray:
    """Deterministic unit-modulus complex phase code for an entity (0 bytes stored)."""
    h = hashlib.blake2b(digest_size=64, key=int(seed).to_bytes(4, "little"))
    h.update(int(entity_id).to_bytes(8, "little", signed=True))
    raw = h.digest()
    ang = (np.frombuffer(raw, dtype=np.uint8)[:D].astype(np.float32) / 255.0) * (2 * np.pi)
    return np.exp(1j * ang)


class CarrierLattice:
    """S cells x D complex int16 carriers."""

    def __init__(self, n_slots: int = 4096, D: int = 16, n_levels: int = 8,
                 seed: int = 0xC0FFEE, superpose: bool = False,
                 snr_threshold: float = 1.5, rep: int = 4):
        self.S, self.D, self.n_levels, self.seed = n_slots, D, n_levels, seed
        self.rep = rep                      # carriers per fact (repetition code)
        self.blocks = max(1, D // rep)      # facts per cell
        self.superpose = superpose
        self.snr_threshold = snr_threshold
        self.cells = np.zeros((n_slots, D, 2), dtype=np.int16)
        self.owner = np.full(n_slots, -1, dtype=np.int64)      # entity id per cell
        self.code_cache: dict[int, np.ndarray] = {}

    # ---------------- addressing ----------------
    def _slot(self, entity_id: int, group: int, coll: int) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(b"carrier")
        h.update(int(entity_id).to_bytes(8, "little", signed=True))
        h.update(int(group).to_bytes(2, "little"))
        h.update(int(coll).to_bytes(1, "little"))
        return int.from_bytes(h.digest(), "little") % self.S

    @staticmethod
    def group_of(rel_id: int, blocks: int) -> int:
        """Which relation-group (spiral page) a relation lives in."""
        return rel_id // blocks

    def code(self, entity_id: int) -> np.ndarray:
        c = self.code_cache.get(entity_id)
        if c is None:
            c = entity_code(entity_id, self.D, self.seed)
            if len(self.code_cache) < 50000:
                self.code_cache[entity_id] = c
        return c

    @staticmethod
    def rel_slot(rel_id: int, D: int) -> int:
        return rel_id % D

    # ---------------- write ----------------
    def write(self, entity_id: int, rel_id: int, obj_ref: int) -> None:
        """Repetition-coded write.

        The relation selects a *block* of `rep` carriers inside the cell; the
        payload is written into every carrier of that block, weighted by the
        entity's phase code.  Reading averages the block (matched filter), so
        carrier noise falls by sqrt(rep) while the payload is preserved.
        """
        if rel_id >= self.blocks * self.rep:
            raise ValueError(f"relation id {rel_id} exceeds capacity "
                             f"{self.blocks * self.rep}")
        u = self.code(entity_id)
        block = self.rel_slot(rel_id, self.blocks)
        payload = complex(obj_ref & 0xFF, (obj_ref >> 8) & 0xFF)
        group = self.group_of(rel_id, self.blocks)
        base = block * self.rep
        for coll in range(self.n_levels):
            slot = self._slot(entity_id, group, coll)
            own = self.owner[slot]
            if own == entity_id or own < 0:
                self.owner[slot] = entity_id
                for k in range(self.rep):
                    self._add(slot, base + k, u[base + k] * payload)
                return
            if self.superpose:
                for k in range(self.rep):
                    self._add(slot, base + k, u[base + k] * payload)
                return
        raise RuntimeError("carrier lattice: all spiral levels occupied")

    def _add(self, slot: int, j: int, v: complex) -> None:
        c = self.cells[slot]
        c[j, 0] = int(np.clip(c[j, 0] + round(v.real), -32767, 32767))
        c[j, 1] = int(np.clip(c[j, 1] + round(v.imag), -32767, 32767))

    # ---------------- read ----------------
    def read(self, entity_id: int, rel_id: int):
        """Returns (slot, wave, payload_bytes, snr) or None.

        Demodulation is a per-carrier phase rotation by conj(u_e): the payload
        lives in the complex plane, the entity's code is its signature, and the
        noise floor is measured from the *inactive* carriers (25th percentile
        of |demodulated|), so "the fact is not there" is a measurement.
        """
        u = self.code(entity_id)
        group = self.group_of(rel_id, self.blocks)
        for coll in range(self.n_levels):
            s = self._slot(entity_id, group, coll)
            if self.owner[s] < 0:
                continue
            if not self.superpose and self.owner[s] != entity_id:
                continue
            cell = self.cells[s].astype(np.float32)
            z = cell[:, 0] + 1j * cell[:, 1]
            dem = np.conj(u) * z
            j = self.rel_slot(rel_id, self.blocks)
            base = j * self.rep
            block = dem[base: base + self.rep]
            half = max(1, self.rep // 2)
            # 2-way interleaved repetition: two independent estimates of the
            # same payload.  If they disagree beyond the noise floor the fact
            # is *damaged* -> abstain (never answer wrongly).
            sig_a = block[:half].sum() / half
            sig_b = block[half:2 * half].sum() / half
            mags = np.abs(dem)
            floor = float(np.percentile(mags, 25)) + 1e-9
            if abs(sig_a - sig_b) > 4.0 * floor + 3.0:
                continue                      # damaged payload -> unknown
            sig = (sig_a + sig_b) / 2.0
            snr = float(abs(sig) / floor)
            if snr < self.snr_threshold:
                continue                      # no usable signal at this carrier
            lo = int(round(sig.real)) & 0xFF
            hi = int(round(sig.imag)) & 0xFF
            obj_ref = lo | (hi << 8)
            payload = bytes([rel_id]) + struct.pack("<H", obj_ref)
            return s, np.array([sig], dtype=np.complex64), payload, snr
        return None

    def iter_facts(self):
        """Decode every stored fact (needs the owner code of each cell)."""
        for s in range(self.S):
            eid = int(self.owner[s])
            if eid < 0:
                continue
            u = self.code(eid)
            cell = self.cells[s].astype(np.float32)
            z = cell[:, 0] + 1j * cell[:, 1]
            dem = np.conj(u) * z
            mags = np.abs(dem)
            floor = float(np.percentile(mags, 25)) + 1e-9
            for b in range(self.blocks):
                base = b * self.rep
                sig = dem[base:base + self.rep].sum() / self.rep
                if abs(sig) < 3 * floor:
                    continue
                lo = int(round(sig.real)) & 0xFF
                hi = int(round(sig.imag)) & 0xFF
                yield eid, b, lo | (hi << 8)

    # ---------------- noise injection ----------------
    def inject_noise(self, sigma: float, rng: np.random.Generator | None = None) -> None:
        """Add Gaussian noise (carrier units) to every occupied cell."""
        rng = rng or np.random.default_rng(0)
        occ = self.owner >= 0
        n = int(occ.sum())
        if n == 0:
            return
        noise = rng.normal(0, sigma, size=(n, self.D, 2))
        self.cells[occ] = np.clip(self.cells[occ].astype(np.float32) + noise,
                                   -32767, 32767).astype(np.int16)

    def load_factor(self) -> float:
        occ = self.owner >= 0
        return float(self.owner[occ].size) / max(1, int(occ.sum()))

    # ---------------- persistence ----------------
    def serialize(self) -> bytes:
        head = struct.pack("<IHBBII", self.S, self.D, self.n_levels,
                           int(self.superpose), int(self.snr_threshold * 1000), self.rep)
        occ = np.nonzero(self.owner >= 0)[0]
        idx = occ.astype(np.uint32).tobytes()
        own = self.owner[occ].astype(np.int64).tobytes()
        data = self.cells[occ].astype(np.int16).tobytes()
        return (head + struct.pack("<I", len(occ)) + idx + own + data)

    @classmethod
    def deserialize(cls, blob: bytes) -> "CarrierLattice":
        S, D, nl, sup, thr, rep = struct.unpack_from("<IHBBII", blob, 0)
        (n,) = struct.unpack_from("<I", blob, 16)
        off = 20
        idx = np.frombuffer(blob, dtype=np.uint32, count=n, offset=off); off += 4 * n
        own = np.frombuffer(blob, dtype=np.int64, count=n, offset=off); off += 8 * n
        data = np.frombuffer(blob, dtype=np.int16, count=n * D * 2, offset=off)
        lat = cls(S, D, nl, superpose=bool(sup), snr_threshold=thr / 1000.0, rep=rep)
        lat.cells[idx] = data.reshape(n, D, 2)
        lat.owner[idx] = own
        return lat
