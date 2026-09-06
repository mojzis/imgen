import json

import pytest

from imgen.cache import Cache, Entry, cache_key
from tests.conftest import PNG


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "cache")


def test_cache_key_is_stable_and_short():
    k = cache_key("google/nano-banana", prompt="a cat", aspect="1:1")
    assert k == cache_key("google/nano-banana", prompt="a cat", aspect="1:1")
    assert len(k) == 16


BASE = cache_key("google/nano-banana", prompt="a cat", aspect="1:1")


@pytest.mark.parametrize(
    "other",
    [
        cache_key("google/nano-banana", prompt="a dog", aspect="1:1"),
        cache_key("google/nano-banana", prompt="a cat", aspect="3:2"),
        cache_key("flux", prompt="a cat", aspect="1:1"),
        cache_key("google/nano-banana", prompt="a cat", aspect="1:1", refs_hash="abc"),
        cache_key("google/nano-banana", prompt="a cat", aspect="1:1", seed=1),
    ],
)
def test_cache_key_changes_with_any_input(other):
    assert other != BASE


def test_cache_key_resolves_model_alias():
    assert cache_key("nano-banana", prompt="a cat", aspect="1:1") == cache_key(
        "google/nano-banana", prompt="a cat", aspect="1:1"
    )


def test_generate_writes_png_and_json(run, cache, ref_png):
    entry = cache.generate("a cat", aspect="3:2", refs=[ref_png], seed=4, tags=["cat"])
    assert entry.png.read_bytes() == PNG
    meta = json.loads(entry.meta_path.read_text())
    assert meta.pop("ts") == entry.ts
    assert meta == {
        "key": entry.key,
        "model": "google/nano-banana",
        "prompt": "a cat",
        "aspect": "3:2",
        "refs": ["ref.png"],
        "seed": 4,
        "price": pytest.approx(0.039),
        "tags": ["cat"],
        "template": None,
        "vars": {},
    }


def test_entry_paths_live_in_cache_dir(run, cache):
    entry = cache.generate("a cat")
    assert (entry.png, entry.meta_path) == (
        cache.dir / f"{entry.key}.png",
        cache.dir / f"{entry.key}.json",
    )


def test_generate_records_template_and_vars(run, cache):
    entry = cache.generate("a red cat", template="cat", vars={"colour": "red"})
    assert (entry.template, entry.vars) == ("cat", {"colour": "red"})


def test_generate_hits_cache_second_time(run, cache):
    first = cache.generate("a cat")
    second = cache.generate("a cat")
    assert first.key == second.key
    assert len(run.calls) == 1


def test_generate_force_regenerates(run, cache):
    cache.generate("a cat")
    cache.generate("a cat", force=True)
    assert len(run.calls) == 2


def test_get_missing_returns_none(cache):
    assert cache.get("0" * 16) is None


def test_get_after_generate(run, cache):
    entry = cache.generate("a cat")
    got = cache.get(entry.key)
    assert got == entry


def test_get_ignores_json_without_png(run, cache):
    entry = cache.generate("a cat")
    entry.png.unlink()
    assert cache.get(entry.key) is None


def test_lookup_by_inputs_without_generating(run, cache, ref_png):
    cache.generate("a cat", refs=[ref_png])
    assert cache.lookup("a cat", refs=[ref_png]) is not None
    assert cache.lookup("a cat") is None
    assert len(run.calls) == 1


def test_entries_lists_all_sorted_newest_first(run, cache):
    a = cache.generate("a")
    b = cache.generate("b")
    cache.touch(b, ts="2026-01-02T00:00:00+00:00")
    cache.touch(a, ts="2026-01-03T00:00:00+00:00")
    assert [e.prompt for e in cache.entries()] == ["a", "b"]


def test_entries_empty_when_dir_missing(cache):
    assert cache.entries() == []


def test_latest_by_tag_prefers_newest(run, cache):
    old = cache.generate("old cat", tags=["cat"])
    new = cache.generate("new cat", tags=["cat", "fresh"])
    cache.generate("a dog", tags=["dog"])
    cache.touch(old, ts="2026-01-01T00:00:00+00:00")
    new = cache.touch(new, ts="2026-01-02T00:00:00+00:00")
    assert cache.latest("cat") == new


def test_latest_unknown_tag_is_none(run, cache):
    cache.generate("a cat", tags=["cat"])
    assert cache.latest("dog") is None


def test_entry_from_meta_roundtrip(run, cache):
    entry = cache.generate("a cat", tags=["x"])
    assert Entry.from_meta(entry.meta_path) == entry


def test_generate_ts_is_utc_isoformat(run, cache):
    entry = cache.generate("a cat")
    assert entry.ts.endswith("+00:00")


def test_get_tolerates_unknown_sidecar_keys(run, cache):
    entry = cache.generate("a cat")
    meta = json.loads(entry.meta_path.read_text())
    meta["from_the_future"] = True
    entry.meta_path.write_text(json.dumps(meta))
    assert cache.get(entry.key) == entry


def test_get_defaults_missing_optional_sidecar_keys(run, cache):
    entry = cache.generate("a cat", tags=["x"])
    meta = json.loads(entry.meta_path.read_text())
    del meta["tags"], meta["template"], meta["vars"]
    entry.meta_path.write_text(json.dumps(meta))
    assert cache.get(entry.key).tags == []
