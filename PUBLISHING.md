# Publishing

`stable-retro-apple-silicon` publishes Apple Silicon wheels for the
`stable-retro` API surface.

## Availability

- PyPI package: `stable-retro-apple-silicon`
- GitHub Releases: matching `.whl` assets for each tagged release
- Target platform: Apple Silicon `arm64`
- Supported macOS baseline: `14.0+`
- Supported Python versions: `3.9` to `3.12`

## Why This Exists

This fork exists to provide a straightforward install path for Apple Silicon
users without asking them to build `stable-retro` from source.

## Versioning

This project tracks upstream `stable-retro` releases and publishes downstream
Apple Silicon fixes as post releases such as `0.9.9.post1`.

## Using The Package

Install from PyPI:

```bash
pip install stable-retro-apple-silicon
```

Or download the wheel directly from GitHub Releases if you want a specific
artifact.

## Release Checklist

1. Update [`/stable_retro/VERSION.txt`](stable_retro/VERSION.txt).
2. Commit and push the release commit.
3. Create a tag such as `v0.9.9.post1`.
4. Publish a GitHub Release for that tag.
5. Let GitHub Actions build the macOS arm64 wheels, publish them to PyPI, and
   attach the `.whl` files to the GitHub Release.
