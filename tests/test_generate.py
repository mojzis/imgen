import httpx
import pytest

import imgen.client as gen_mod
from imgen import GenerationError, generate, generate_to, have_token, prompt_note
from tests.conftest import PNG, FakeFileOutput


def test_have_token_false_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    assert not have_token()


def test_have_token_true_when_set(monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_x")
    assert have_token()


def test_have_token_loads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("REPLICATE_API_TOKEN=r8_from_file\n")
    assert have_token()


def test_generate_without_token_raises(monkeypatch, tmp_path, run):
    monkeypatch.delenv("REPLICATE_API_TOKEN")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(GenerationError, match="REPLICATE_API_TOKEN"):
        generate("p")
    assert run.calls == []


def test_generate_returns_bytes_from_file_output(run):
    assert generate("a cat") == PNG


def test_generate_calls_default_model_with_shaped_input(run):
    generate("a cat", aspect="3:2")
    assert run.calls == [
        ("google/nano-banana", {"prompt": "a cat", "aspect_ratio": "3:2"})
    ]


def test_generate_resolves_model_alias(run):
    generate("a cat", model="flux")
    assert run.calls[0][0] == "black-forest-labs/flux-1.1-pro"


def test_generate_unwraps_list_output(run):
    run.output = [FakeFileOutput(b"first"), FakeFileOutput(b"second")]
    assert generate("p") == b"first"


def test_generate_fetches_url_output(run, monkeypatch):
    run.output = "https://example.test/out.png"
    seen = {}

    def fake_get(url, timeout):
        seen["url"] = url
        return httpx.Response(
            200, content=b"from-url", request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(gen_mod.httpx, "get", fake_get)
    assert generate("p") == b"from-url"
    assert seen["url"] == "https://example.test/out.png"


def test_generate_empty_output_raises(run):
    run.output = FakeFileOutput(b"")
    with pytest.raises(GenerationError, match="no image"):
        generate("p")


def test_generate_empty_list_raises(run):
    run.output = []
    with pytest.raises(GenerationError, match="no image"):
        generate("p")


def test_generate_wraps_replicate_errors(run):
    run.output = RuntimeError("boom from replicate")
    with pytest.raises(GenerationError, match="boom from replicate"):
        generate("p")


def test_generate_options_override_shaped_input(run):
    generate("p", options={"aspect_ratio": "9:16", "extra": 1})
    assert run.last_input == {"prompt": "p", "aspect_ratio": "9:16", "extra": 1}


def test_generate_passes_refs_and_seed(run, ref_png):
    generate("p", refs=[ref_png], seed=3)
    assert run.last_input["seed"] == 3
    assert len(run.last_input["image_input"]) == 1


def test_generate_to_writes_png_and_sidecar(run, tmp_path, ref_png):
    out = tmp_path / "out" / "cat.png"
    result = generate_to(out, "a cat", aspect="3:2", refs=[ref_png], seed=5)
    assert result == out
    assert out.read_bytes() == PNG
    note = out.with_suffix(".prompt.txt").read_text()
    assert note == (
        "# model: google/nano-banana\n# aspect: 3:2\n# refs: ref.png\n# seed: 5\n\na cat\n"
    )


def test_generate_to_sidecar_minimal(run, tmp_path):
    out = tmp_path / "cat.png"
    generate_to(out, "a cat")
    assert out.with_suffix(".prompt.txt").read_text() == (
        "# model: google/nano-banana\n# aspect: 1:1\n\na cat\n"
    )


def test_generate_to_skips_existing(run, tmp_path):
    out = tmp_path / "cat.png"
    out.write_bytes(b"old")
    generate_to(out, "a cat")
    assert out.read_bytes() == b"old"
    assert run.calls == []


def test_generate_to_force_overwrites(run, tmp_path):
    out = tmp_path / "cat.png"
    out.write_bytes(b"old")
    generate_to(out, "a cat", force=True)
    assert out.read_bytes() == PNG


def test_generate_url_error_status_raises(run, monkeypatch):
    run.output = "https://example.test/expired.png"
    req = httpx.Request("GET", run.output)
    monkeypatch.setattr(
        gen_mod.httpx,
        "get",
        lambda url, timeout: httpx.Response(
            404, content=b"<html>gone</html>", request=req
        ),
    )
    with pytest.raises(GenerationError, match="404"):
        generate("p")


def test_generate_url_transport_error_raises(run, monkeypatch):
    run.output = "https://example.test/out.png"

    def boom(url, timeout):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(gen_mod.httpx, "get", boom)
    with pytest.raises(GenerationError, match="no route"):
        generate("p")


def test_generate_unexpected_output_type_raises(run):
    run.output = {"weird": 1}
    with pytest.raises(GenerationError, match="unexpected model output: dict"):
        generate("p")


def test_prompt_note_is_prompt_first_keyword_rest():
    assert (
        prompt_note("a cat", model="m", aspect="1:1")
        == "# model: m\n# aspect: 1:1\n\na cat\n"
    )
