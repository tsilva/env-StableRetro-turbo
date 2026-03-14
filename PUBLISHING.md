# Publishing

This fork publishes public Apple Silicon wheels for
`stable-retro-apple-silicon`.

## PyPI Continuity

Versions `0.9.10` through `0.9.12` were already published for this package from
the earlier transitional repo layout. This fork now becomes the canonical source
tree, so the next public release from here should be `0.9.13` or newer.

## Before The First Release From This Fork

1. Create the PyPI trusted publisher entry for this GitHub repo and
   [`/.github/workflows/release.yml`](.github/workflows/release.yml).
2. Confirm the default branch on GitHub is either `master` or `main`.

## Release Checklist

1. Update [`/stable_retro/VERSION.txt`](stable_retro/VERSION.txt).
2. Commit and push the release commit.
3. Create a tag such as `v0.9.13`.
4. Publish a GitHub Release for that tag.
5. Let GitHub Actions build the macOS arm64 wheels, publish them to PyPI, and
   attach the `.whl` files to the GitHub Release.

## Local Verification

Build the public wheel locally:

```bash
python -m pip install --upgrade build
CMAKE_ARGS="-DBUILD_CORES=gb;nes;snes;genesis -DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF" \
STABLE_RETRO_PUBLIC_CORES="gambatte,fceumm,snes9x,genesis_plus_gx" \
python -m build --wheel
```

Inspect that the wheel contains no `rom.*` payloads except `rom.sha`:

```bash
unzip -l dist/*.whl | rg 'rom\.'
```

Expected result:

- `rom.sha` entries may appear
- actual ROM payloads such as `rom.gb`, `rom.nes`, `rom.sfc`, and `rom.sms`
  must not appear
