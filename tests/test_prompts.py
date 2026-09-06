import pytest

from imgen import PromptError
from imgen.prompts import render, render_file


def test_render_substitutes_placeholders():
    assert render("A {animal} on a {thing}.", animal="cat", thing="mat") == (
        "A cat on a mat."
    )


def test_render_missing_variable_names_it():
    with pytest.raises(PromptError, match="animal"):
        render("A {animal}.", thing="mat")


def test_render_strips_surrounding_whitespace():
    assert render("\n  A {x}.\n\n", x="cat") == "A cat."


def test_render_keeps_literal_braces():
    assert render("{{not a var}} {x}", x="y") == "{not a var} y"


def test_render_file_reads_template(tmp_path):
    p = tmp_path / "cat.txt"
    p.write_text("A {colour} cat.\n")
    assert render_file(p, colour="red") == "A red cat."


def test_render_file_error_mentions_path(tmp_path):
    p = tmp_path / "cat.txt"
    p.write_text("A {colour} cat.\n")
    with pytest.raises(PromptError, match=r"cat\.txt"):
        render_file(p)


def test_render_allows_template_as_variable_name():
    assert render("{template} {path}", template="a", path="b") == "a b"


def test_render_file_allows_path_as_variable_name(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("{path}")
    assert render_file(p, path="x") == "x"


@pytest.mark.parametrize("template", ["{ broken", "{0}", "}"])
def test_render_malformed_template_is_prompt_error(template):
    with pytest.raises(PromptError, match="bad template"):
        render(template)
