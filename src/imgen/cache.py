"""Exact-prompt cache: a png + json sidecar per (model, prompt, aspect, refs, seed)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from imgen.client import generate
from imgen.models import DEFAULT_MODEL, resolve_model
from imgen.refs import refs_digest

KEY_LEN = 16


def cache_key(
    model: str,
    *,
    prompt: str,
    aspect: str,
    refs_hash: str = "",
    seed: int | None = None,
) -> str:
    """sha256 of the inputs that change the picture, truncated to 16 hex chars."""
    parts = [
        resolve_model(model).id,
        prompt,
        aspect,
        refs_hash,
        "" if seed is None else str(seed),
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:KEY_LEN]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Entry:
    """What the json sidecar holds; ``png`` and ``meta_path`` derive from dir + key."""

    dir: pathlib.Path
    key: str
    model: str
    prompt: str
    aspect: str
    refs: list[str]
    seed: int | None
    ts: str
    price: float
    tags: list[str] = field(default_factory=list)
    template: str | None = None
    vars: dict[str, Any] = field(default_factory=dict)

    @property
    def png(self) -> pathlib.Path:
        return self.dir / f"{self.key}.png"

    @property
    def meta_path(self) -> pathlib.Path:
        return self.dir / f"{self.key}.json"

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        del d["dir"]
        return d

    @classmethod
    def from_meta(cls, path: pathlib.Path) -> Entry:
        """Unknown keys (a newer imgen) are ignored; missing optional ones default."""
        known = {f.name for f in fields(cls)} - {"dir"}
        raw = json.loads(path.read_text())
        return cls(dir=path.parent, **{k: v for k, v in raw.items() if k in known})

    def write_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(self.to_meta(), indent=2, ensure_ascii=False)
        )


class Cache:
    """A directory of ``<key>.png`` + ``<key>.json``.

    Holds raw model output only; post-processed variants belong elsewhere.
    """

    def __init__(self, dir: pathlib.Path) -> None:
        self.dir = dir

    def get(self, key: str) -> Entry | None:
        meta = self.dir / f"{key}.json"
        if not meta.exists() or not (self.dir / f"{key}.png").exists():
            return None
        return Entry.from_meta(meta)

    def lookup(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        aspect: str = "1:1",
        refs: Sequence[pathlib.Path] = (),
        seed: int | None = None,
    ) -> Entry | None:
        """The entry these inputs would produce, if it is already cached."""
        return self.get(
            cache_key(
                model,
                prompt=prompt,
                aspect=aspect,
                refs_hash=refs_digest(refs),
                seed=seed,
            )
        )

    def generate(  # noqa: PLR0913 -- keyword-only knobs, mirrors imgen.generate
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        aspect: str = "1:1",
        refs: Sequence[pathlib.Path] = (),
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        tags: Sequence[str] = (),
        template: str | None = None,
        vars: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Entry:
        """Return the cached entry for these inputs, generating it first if needed."""
        key = cache_key(
            model, prompt=prompt, aspect=aspect, refs_hash=refs_digest(refs), seed=seed
        )
        if not force and (hit := self.get(key)) is not None:
            return hit
        m = resolve_model(model)
        blob = generate(
            prompt, model=model, aspect=aspect, refs=refs, seed=seed, options=options
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = Entry(
            dir=self.dir,
            key=key,
            model=m.id,
            prompt=prompt,
            aspect=aspect,
            refs=[r.name for r in refs],
            seed=seed,
            ts=_now(),
            price=m.price,
            tags=list(tags),
            template=template,
            vars=dict(vars or {}),
        )
        entry.png.write_bytes(blob)
        entry.write_meta()
        return entry

    def touch(self, entry: Entry, ts: str | None = None) -> Entry:
        """Rewrite the sidecar with a new timestamp (``now`` by default)."""
        fresh = replace(entry, ts=ts or _now())
        fresh.write_meta()
        return fresh

    def entries(self) -> list[Entry]:
        """Every complete entry, newest first."""
        if not self.dir.is_dir():
            return []
        found = [Entry.from_meta(p) for p in self.dir.glob(f"{'?' * KEY_LEN}.json")]
        return sorted(
            (e for e in found if e.png.exists()), key=lambda e: e.ts, reverse=True
        )

    def latest(self, tag: str) -> Entry | None:
        """Newest entry carrying ``tag``: the latest picture for a subject."""
        return next((e for e in self.entries() if tag in e.tags), None)
