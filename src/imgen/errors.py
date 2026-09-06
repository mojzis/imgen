"""Exceptions raised by imgen."""


class ImgenError(Exception):
    """Base class for every imgen error."""


class GenerationError(ImgenError):
    """The model call failed, returned nothing, or could not be made."""


class PromptError(ImgenError):
    """A prompt template could not be rendered."""


class StyleError(ImgenError):
    """A style file or entry is malformed."""


class ModelError(ImgenError):
    """A model alias is not known."""
