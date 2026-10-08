"""Table-driven register decoding.

Everything about the wire layout (addresses, widths, word order, signedness,
scale, unit) comes from ``config/register_map.yaml``; nothing is hard-coded
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import DEFAULT_REGISTER_MAP_FILE, load_yaml

TIME_MEASUREMENT = "t_sim"
_EPOCH = datetime(1970, 1, 1)


class RegisterDef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    measurement: str
    phase: Optional[Literal["a", "b", "c", "total"]] = None
    function_code: Literal[3, 4]
    address: int = Field(ge=0)
    width_registers: Literal[1, 2, 4]
    word_order: Literal["big", "little"] = "big"
    signed: bool = False
    scale: float = 1.0
    raw_unit: Optional[str] = None
    unit: str
    obis: Optional[str] = None

    def decode_raw(self, words: list[int]) -> int:
        """Combine 16-bit words into an integer (word order + signedness)."""
        ws = list(words) if self.word_order == "big" else list(reversed(words))
        value = 0
        for w in ws:
            value = (value << 16) | (w & 0xFFFF)
        if self.signed:
            bits = 16 * self.width_registers
            if value >= 1 << (bits - 1):
                value -= 1 << bits
        return value

    def decode(self, words: list[int]) -> float:
        """Scaled value, rounded to the register's quantum (e.g. 0.01 V) so
        float artefacts like 6006.1900000000005 do not leak downstream."""
        value = self.decode_raw(words) * self.scale
        return round(value, self.decimals) if self.decimals else float(value)

    @property
    def decimals(self) -> int:
        d, s = 0, abs(self.scale)
        while s and s < 1 and d < 12:
            s *= 10
            d += 1
        return d


@dataclass(frozen=True)
class ReadBlock:
    function_code: int
    start: int          # zero-based offset (add address_base on the wire)
    count: int


class RegisterMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_base: int = 0
    max_registers_per_read: int = Field(125, ge=1, le=125)
    registers: list[RegisterDef]

    @model_validator(mode="after")
    def _no_overlap(self) -> "RegisterMap":
        seen: dict[tuple[int, int], str] = {}
        for r in self.registers:
            for off in range(r.address, r.address + r.width_registers):
                key = (r.function_code, off)
                if key in seen:
                    raise ValueError(f"{r.name} overlaps {seen[key]} at fc{key[0]}:{off}")
                seen[key] = r.name
        names = [r.name for r in self.registers]
        if len(set(names)) != len(names):
            raise ValueError("duplicate register names")
        return self

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGISTER_MAP_FILE) -> "RegisterMap":
        return cls.model_validate(load_yaml(path))

    @property
    def time_register(self) -> Optional[RegisterDef]:
        for r in self.registers:
            if r.measurement == TIME_MEASUREMENT:
                return r
        return None

    def wire_address(self, offset: int) -> int:
        return offset + self.address_base

    def mapped_offsets(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for r in self.registers:
            out.setdefault(r.function_code, set()).update(
                range(r.address, r.address + r.width_registers))
        return out

    def blocks(self, gap_fill: int = 0) -> list[ReadBlock]:
        """Contiguous read blocks per function code, each ≤ max_registers_per_read.

        Registers separated by at most ``gap_fill`` unmapped words are merged
        into one read.
        """
        blocks: list[ReadBlock] = []
        for fc in sorted({r.function_code for r in self.registers}):
            spans = sorted((r.address, r.address + r.width_registers)
                           for r in self.registers if r.function_code == fc)
            cur_s, cur_e = spans[0]
            for s, e in spans[1:]:
                if s <= cur_e + gap_fill and max(e, cur_e) - cur_s <= self.max_registers_per_read:
                    cur_e = max(cur_e, e)
                else:
                    blocks.extend(self._split(fc, cur_s, cur_e))
                    cur_s, cur_e = s, e
            blocks.extend(self._split(fc, cur_s, cur_e))
        return blocks

    def _split(self, fc: int, start: int, end: int) -> Iterable[ReadBlock]:
        step = self.max_registers_per_read
        s = start
        while s < end:
            n = min(step, end - s)
            yield ReadBlock(fc, s, n)
            s += n

    # ------------------------------------------------------------------ #
    def decode_block_values(
        self, raw: dict[int, dict[int, int]]
    ) -> dict[str, float]:
        """Decode every register whose words are all present in ``raw``.

        ``raw`` maps function_code -> {offset: word}.  Registers with missing
        words (failed block) are left out.
        """
        out: dict[str, float] = {}
        for r in self.registers:
            words_by_off = raw.get(r.function_code, {})
            offs = range(r.address, r.address + r.width_registers)
            if all(o in words_by_off for o in offs):
                out[r.name] = r.decode([words_by_off[o] for o in offs])
        return out

    def build_measurements(
        self, values: dict[str, float], phases_present: Iterable[str]
    ) -> dict:
        """Group decoded values into the payload ``measurements`` object.

        Phases not in ``phases_present`` are omitted.  ``total`` is the sum of
        the present phases, added only when every present phase was read and
        the map does not serve a total register itself.
        """
        phases = set(phases_present)
        grouped: dict[str, dict[str, float] | float] = {}
        for r in self.registers:
            if r.measurement == TIME_MEASUREMENT or r.name not in values:
                continue
            v = values[r.name]
            if r.phase is None:
                grouped[r.measurement] = v
            elif r.phase == "total" or r.phase in phases:
                grouped.setdefault(r.measurement, {})[r.phase] = v  # type: ignore[index]
        decimals: dict[str, int] = {}
        for r in self.registers:
            decimals[r.measurement] = max(decimals.get(r.measurement, 0), r.decimals)
        for key in ("active_power_kw", "reactive_power_kvar"):
            d = grouped.get(key)
            if isinstance(d, dict) and d and "total" not in d and set(d) >= phases:
                d["total"] = round(sum(d[p] for p in sorted(phases)), decimals.get(key, 6))
        return grouped

    def decode_time(self, values: dict[str, float]) -> Optional[datetime]:
        """Simulated time as a naive datetime; ``None`` if absent or still 0."""
        tr = self.time_register
        if tr is None or tr.name not in values:
            return None
        secs = values[tr.name]
        if secs <= 0:
            return None
        return _EPOCH + timedelta(seconds=secs)
