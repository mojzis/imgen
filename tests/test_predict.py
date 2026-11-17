import httpx
import pytest

import imgen.client as gen_mod
from imgen import GenerationError

GET = "https://api.replicate.com/v1/predictions/p1"


@pytest.fixture
def api(monkeypatch):
    """Replicate's API faked: `answers` are returned in order; requests recorded."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")
    monkeypatch.setattr(gen_mod, "POLL_SECONDS", 0)
    state = {"answers": [], "seen": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["seen"].append((request.method, str(request.url)))
        status, body = state["answers"].pop(0)
        return httpx.Response(status, json=body)

    real = httpx.Client
    monkeypatch.setattr(
        gen_mod.httpx,
        "Client",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return state


def pred(status: str, **more) -> dict:
    return {"id": "p1", "version": None, "status": status, "urls": {"get": GET}, **more}


def test_polls_an_official_model_until_it_succeeds(api):
    api["answers"] = [
        (201, pred("starting")),
        (200, pred("succeeded", output="out-1")),
    ]

    out = gen_mod._predict("google/nano-banana", input={"prompt": "a cat"})

    assert out == "out-1"
    assert api["seen"] == [
        ("POST", "https://api.replicate.com/v1/models/google/nano-banana/predictions"),
        ("GET", GET),
    ]


def test_a_pinned_version_goes_to_predictions(api):
    api["answers"] = [(201, pred("succeeded", output=["out-2"]))]

    gen_mod._predict("owner/name:abc123", input={})

    assert api["seen"] == [("POST", "https://api.replicate.com/v1/predictions")]


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ((201, pred("failed", error="NSFW")), "failed: NSFW"),
        ((422, {"detail": "bad input"}), "422.*bad input"),
    ],
)
def test_failures_raise_generation_error(api, answer, message):
    api["answers"] = [answer]

    with pytest.raises(GenerationError, match=message):
        gen_mod._predict("google/nano-banana", input={})
