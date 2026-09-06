import pytest

from imgen import StyleError
from imgen.style import Style, compose, load_style, load_styles

STYLES_YAML = """\
# comment
storybook:
  style: >
    Children's storybook illustration, warm.
  palette: ["#f3ead4", "#e9b959"]
  mood: warm, playful
  negative: No text, no letters.
  aspect_ratio: "1:1"
ikona:
  style: Byzantine icon.
  aspect_ratio: "2:3"
  coloring: Trace it.
  model: flux
"""

THEME_YAML = """\
style: Hand-drawn comic.
palette:
  - "#f4d35e"
mood: warm
negative: No text.
aspect_ratio: "1:1"
"""


@pytest.fixture
def styles_file(tmp_path):
    p = tmp_path / "styles.yaml"
    p.write_text(STYLES_YAML)
    return p


def test_load_styles_returns_mapping_of_styles(styles_file):
    styles = load_styles(styles_file)
    assert set(styles) == {"storybook", "ikona"}
    assert styles["storybook"] == Style(
        style="Children's storybook illustration, warm.\n",
        palette=["#f3ead4", "#e9b959"],
        mood="warm, playful",
        negative="No text, no letters.",
        aspect_ratio="1:1",
    )


def test_load_styles_optional_fields(styles_file):
    ikona = load_styles(styles_file)["ikona"]
    assert (ikona.coloring, ikona.model, ikona.palette, ikona.mood) == (
        "Trace it.",
        "flux",
        [],
        "",
    )


def test_load_style_single_theme_file(tmp_path):
    p = tmp_path / "theme.yaml"
    p.write_text(THEME_YAML)
    assert load_style(p) == Style(
        style="Hand-drawn comic.",
        palette=["#f4d35e"],
        mood="warm",
        negative="No text.",
        aspect_ratio="1:1",
    )


def test_style_from_dict_rejects_unknown_keys():
    with pytest.raises(StyleError, match="unknown style key"):
        Style.from_dict({"style": "x", "moood": "y"})


def test_style_from_dict_requires_style():
    with pytest.raises(StyleError, match="'style'"):
        Style.from_dict({"mood": "y"})


def test_load_styles_error_names_file_and_id(tmp_path):
    p = tmp_path / "styles.yaml"
    p.write_text("ikona:\n  mood: x\n")
    with pytest.raises(StyleError, match=r"styles\.yaml: style 'ikona': missing"):
        load_styles(p)


def test_load_style_error_names_file(tmp_path):
    p = tmp_path / "theme.yaml"
    p.write_text("bogus: 1\n")
    with pytest.raises(StyleError, match=r"theme\.yaml: unknown"):
        load_style(p)


def test_style_default_aspect_is_square():
    assert Style(style="x").aspect_ratio == "1:1"


FULL = Style(
    style="  Storybook.  ",
    palette=["#111", "#222"],
    mood="warm",
    negative="No text.",
)


def test_compose_full_style():
    assert compose(FULL, "  a cat  ") == (
        "Storybook.\n\nSubject: a cat\n\n"
        "Use this color palette: #111, #222.\n\nMood: warm.\n\nNo text."
    )


def test_compose_minimal_style():
    assert compose(Style(style="Ink."), "a cat") == "Ink.\n\nSubject: a cat"


def test_compose_extra_negatives_are_joined():
    out = compose(
        Style(style="Ink.", negative="No text."), "a cat", ["No hats.", "", " "]
    )
    assert out.endswith("\n\nNo text. No hats.")


def test_compose_extra_negatives_without_style_negative():
    out = compose(Style(style="Ink."), "a cat", ["No hats."])
    assert out == "Ink.\n\nSubject: a cat\n\nNo hats."
