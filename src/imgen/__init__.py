"""imgen — small shared library for generating images through Replicate."""

from importlib.metadata import version

from imgen.cache import Cache, Entry, cache_key
from imgen.client import generate, generate_to, have_token, prompt_note
from imgen.errors import (
    GenerationError,
    ImgenError,
    ModelError,
    PromptError,
    StyleError,
)
from imgen.models import DEFAULT_MODEL, MODELS, Model, price, resolve_model
from imgen.prompts import render, render_file
from imgen.refs import data_uri, refs_digest
from imgen.style import Style, compose, load_style, load_styles

__version__ = version("imgen")

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "Cache",
    "Entry",
    "GenerationError",
    "ImgenError",
    "Model",
    "ModelError",
    "PromptError",
    "Style",
    "StyleError",
    "__version__",
    "cache_key",
    "compose",
    "data_uri",
    "generate",
    "generate_to",
    "have_token",
    "load_style",
    "load_styles",
    "price",
    "prompt_note",
    "refs_digest",
    "render",
    "render_file",
    "resolve_model",
]
