# Changelog

## Unreleased

- Predictions go through Replicate's HTTP API with httpx; the `replicate` dependency is gone. replicate 1.0.7 rejected the API's `"version": null` for official models (nano-banana etc.), so every `generate` failed with `none is not an allowed value`.
