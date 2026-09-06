# imgen

Small shared library for generating images through Replicate.

## Install

Not on PyPI — install a tagged release from GitHub:

```
uv add git+https://github.com/mojzis/imgen --tag vX.Y.Z
```

PEP 723 script: `"imgen @ git+https://github.com/mojzis/imgen@vX.Y.Z"`.

Set `REPLICATE_API_TOKEN` in the environment or a `.env` in the working directory.

## Usage

### One image

```python
from pathlib import Path

from imgen import generate, generate_to

png = generate("a cat on a mat", model="nano-banana", aspect="3:2")

# writes cat.png + cat.prompt.txt (model, aspect, refs, seed, prompt); skips if it exists
generate_to(
    Path("out/cat.png"), "a cat on a mat", refs=[Path("refs/cat.png")], force=False
)
```

Models: `nano-banana` (default), `seedream`, `flux`, `imagen-3`, `sdxl-lineart`, or any
Replicate ref. Aliases resolve to full ids; unknown refs get the `image_input` shape.
`price(model)` gives the USD list price per image. `options={...}` merges model-specific
knobs into the shaped input.

### Styles and templates

```python
from imgen import load_styles, compose, render_file

styles = load_styles(Path("styles.yaml"))  # id -> Style
prompt = compose(styles["storybook"], "St. Wenceslas feeding the poor", ["No halo."])
prompt = render_file(Path("prompts/word.txt"), word="apple")
```

`styles.yaml` entries: `style`, `palette`, `mood`, `negative`, `aspect_ratio`, optional
`coloring` and `model`. A single-style `theme.yaml` loads with `load_style`.

### Cache

```python
from imgen import Cache

cache = Cache(Path("cache"))
entry = cache.generate(
    prompt,
    aspect=style.aspect_ratio,
    tags=["vaclav"],
    template="saint",
    vars={"name": "Vaclav"},
)
entry.png  # cache/<key>.png, raw model output
entry.meta_path  # cache/<key>.json: model, prompt, aspect, refs, seed, ts, price, tags, template, vars
cache.latest("vaclav")  # newest entry with that tag
cache.lookup(prompt, aspect="1:1")  # exact-input hit without generating
```

Key = sha of (model, prompt, aspect, refs contents, seed). `force=True` regenerates.
Post-processing (coloring, background cleanup, composition) stays in the consumer.
