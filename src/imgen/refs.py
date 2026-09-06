"""Reference images: data-URI encoding and content digests."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import pathlib
from collections.abc import Iterable


def data_uri(path: pathlib.Path) -> str:
    """Inline an image file as a ``data:`` URI (Replicate accepts it as image input)."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def refs_digest(paths: Iterable[pathlib.Path]) -> str:
    """Order-sensitive sha256 over the *contents* of the reference images.

    Empty string for no refs, so cache keys stay stable for the common case.
    """
    h = hashlib.sha256()
    any_ref = False
    for p in paths:
        any_ref = True
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest() if any_ref else ""
