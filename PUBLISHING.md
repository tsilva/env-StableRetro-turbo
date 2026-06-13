# Publishing

`stable-retro-turbo` publishes macOS Apple Silicon and Linux wheels for the
`stable-retro` API surface.

## Availability

- PyPI package: `stable-retro-turbo`
- GitHub Releases: matching `.whl` assets for each tagged release
- Target platforms: macOS Apple Silicon `arm64` and Linux `x86_64`
- Supported macOS baseline: `14.0+`
- Supported Python version: `3.14`

## Why This Exists

This fork exists to provide a straightforward install path for Apple Silicon and
Linux users without asking them to build `stable-retro` and the bundled public
cores from source.

## Versioning

This project tracks the upstream `stable-retro` version and publishes downstream
patches as PEP 440 post releases:

```text
<upstream stable-retro version>.post<stable-retro-turbo patch number>
```

For example, if upstream `stable-retro` is `1.0.0` and this fork's patch number
is `1`, publish `1.0.0.post1`.

Use `.postN`, not a local version such as `+apple.N`, because PyPI and pip handle
post releases cleanly for public packages. The old `stable-retro-apple-silicon`
distribution used its own counter; `stable-retro-turbo` starts again at
`1.0.0.post1`.

## Using The Package

Install from PyPI:

```bash
pip install stable-retro-turbo
```

Or download the wheel directly from GitHub Releases if you want a specific
artifact.

## Release Checklist

1. Check the latest upstream `stable-retro` version on PyPI.
2. Update [`/stable_retro/VERSION.txt`](stable_retro/VERSION.txt) using the
   `<upstream>.post<N>` scheme.
3. Commit and push the release commit.
4. Create a tag such as `v1.0.0.post1`.
5. Publish a GitHub Release for that tag.
6. Let GitHub Actions build the macOS arm64 and Linux x86_64 wheels, publish
   them to PyPI, and attach the `.whl` files to the GitHub Release.
