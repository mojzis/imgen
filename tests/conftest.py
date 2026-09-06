from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import pytest

import imgen.client as gen_mod

PNG = b"\x89PNG\r\n\x1a\nfake"


class FakeFileOutput:
    def __init__(self, blob: bytes = PNG) -> None:
        self.blob = blob

    def read(self) -> bytes:
        return self.blob


@dataclass
class FakeRun:
    """Stand-in for replicate.run: records calls, returns a canned output."""

    output: Any = field(default_factory=FakeFileOutput)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, ref: str, input: dict[str, Any]) -> Any:
        self.calls.append((ref, input))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output

    @property
    def last_input(self) -> dict[str, Any]:
        return self.calls[-1][1]


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(gen_mod, "_run", fake)
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")
    return fake


@pytest.fixture
def ref_png(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "ref.png"
    p.write_bytes(PNG)
    return p
