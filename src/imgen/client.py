"""The one call that costs money: shape input, run the model, unwrap the output."""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv

from imgen.errors import GenerationError
from imgen.models import DEFAULT_MODEL, resolve_model

TOKEN_VAR = "REPLICATE_API_TOKEN"  # noqa: S105 -- variable name, not a secret
URL_TIMEOUT = 120
API = "https://api.replicate.com/v1/"
POLL_SECONDS = 2
RUN_TIMEOUT = 600  # Seconds a prediction may take before we give up
RUNNING = {"starting", "processing"}


def _check(resp: httpx.Response) -> dict[str, Any]:
    if resp.is_error:
        raise GenerationError(f"Replicate answered {resp.status_code}: {resp.text}")
    return resp.json()


def _predict(model_id: str, *, input: dict[str, Any]) -> Any:
    """Run one prediction through Replicate's HTTP API and return its output.

    ``model_id`` is ``owner/name`` (an official model, its latest version) or
    ``owner/name:version``. Replaces ``replicate.run``: replicate 1.0.7 rejects the
    API's ``"version": null`` for official models.
    """
    name, _, version = model_id.partition(":")
    if version:
        url, body = f"{API}predictions", {"version": version, "input": input}
    else:
        url, body = f"{API}models/{name}/predictions", {"input": input}
    headers = {"Authorization": f"Bearer {os.environ[TOKEN_VAR]}", "Prefer": "wait"}
    deadline = time.monotonic() + RUN_TIMEOUT
    with httpx.Client(headers=headers, timeout=URL_TIMEOUT) as http:
        pred = _check(http.post(url, json=body))
        while pred["status"] in RUNNING:
            if time.monotonic() > deadline:
                raise GenerationError(f"prediction {pred['id']} timed out")
            time.sleep(POLL_SECONDS)
            pred = _check(http.get(pred["urls"]["get"]))
    if pred["status"] != "succeeded":
        raise GenerationError(f"prediction {pred['status']}: {pred.get('error')}")
    return pred["output"]


_run = _predict  # indirection so tests can swap the network call


def have_token() -> bool:
    """True when a Replicate token is set, loading ``.env`` from the cwd first."""
    if not os.getenv(TOKEN_VAR):
        load_dotenv(find_dotenv(usecwd=True))
    return bool(os.getenv(TOKEN_VAR))


def _fetch(url: str) -> bytes:
    try:
        resp = httpx.get(url, timeout=URL_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GenerationError(f"could not fetch model output: {e}") from e
    return resp.content


def _unwrap(out: Any) -> bytes:
    if isinstance(out, list):
        if not out:
            raise GenerationError("the model returned no image")
        out = out[0]
    if hasattr(out, "read"):
        blob = out.read()
    elif isinstance(out, str):
        blob = _fetch(out)
    else:
        raise GenerationError(f"unexpected model output: {type(out).__name__}")
    if not blob:
        raise GenerationError("the model returned no image")
    return blob


def generate(  # noqa: PLR0913 -- keyword-only knobs
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    aspect: str = "1:1",
    refs: Sequence[pathlib.Path] = (),
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> bytes:
    """Generate one image and return its bytes.

    ``options`` is merged last into the shaped input, for model-specific knobs.
    """
    if not have_token():
        raise GenerationError(f"{TOKEN_VAR} is not set (put it in .env)")
    m = resolve_model(model)
    inp = m.shape(prompt, aspect, refs, seed)
    if options:
        inp.update(options)
    try:
        out = _run(m.id, input=inp)
    except Exception as e:
        raise GenerationError(str(e)) from e
    return _unwrap(out)


def prompt_note(
    prompt: str,
    *,
    model: str,
    aspect: str,
    refs: Sequence[pathlib.Path] = (),
    seed: int | None = None,
) -> str:
    """The ``.prompt.txt`` sidecar: header comments, blank line, the prompt."""
    head = [f"# model: {model}", f"# aspect: {aspect}"]
    if refs:
        head.append("# refs: " + ", ".join(r.name for r in refs))
    if seed is not None:
        head.append(f"# seed: {seed}")
    return "\n".join(head) + "\n\n" + prompt + "\n"


def generate_to(  # noqa: PLR0913 -- keyword-only knobs
    path: pathlib.Path,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    aspect: str = "1:1",
    refs: Sequence[pathlib.Path] = (),
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    force: bool = False,
) -> pathlib.Path:
    """Write the image to ``path`` plus a ``.prompt.txt`` sidecar next to it.

    Skips the call when ``path`` already exists, unless ``force``.
    """
    if path.exists() and not force:
        return path
    blob = generate(
        prompt, model=model, aspect=aspect, refs=refs, seed=seed, options=options
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    path.with_suffix(".prompt.txt").write_text(
        prompt_note(
            prompt, model=resolve_model(model).id, aspect=aspect, refs=refs, seed=seed
        )
    )
    return path
