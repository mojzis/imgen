import base64

from imgen.refs import data_uri, refs_digest


def test_data_uri_png(ref_png):
    uri = data_uri(ref_png)
    head, _, payload = uri.partition(",")
    assert head == "data:image/png;base64"
    assert base64.b64decode(payload) == ref_png.read_bytes()


def test_data_uri_falls_back_to_png_mime(tmp_path):
    p = tmp_path / "noext"
    p.write_bytes(b"x")
    assert data_uri(p).startswith("data:image/png;base64,")


def test_data_uri_jpeg(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"x")
    assert data_uri(p).startswith("data:image/jpeg;base64,")


def test_refs_digest_empty_is_empty_string():
    assert refs_digest([]) == ""


def test_refs_digest_depends_on_content_not_name(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert refs_digest([a]) == refs_digest([b])


def test_refs_digest_is_order_sensitive(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    assert refs_digest([a, b]) != refs_digest([b, a])
