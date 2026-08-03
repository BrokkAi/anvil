# @brokkai/anvil npm packaging

This directory builds the public `@brokkai/anvil` launcher and five distinct
native platform packages. The root package exposes the released Rust `anvil`
binary; it does not download a product binary at first run.

Platform payloads use separate stable package names such as
`@brokkai/anvil-linux-x64`. This keeps every platform package independent from
the root package's `latest` tag. Publish and verify every platform package
before publishing the root package.

Run packaging tests with:

```bash
npm --prefix npm test
```

Build tarballs from extracted release bundles with:

```bash
node npm/scripts/build-packages.mjs \
  --version 0.24.2 \
  --output-dir dist/npm \
  --repository-root /path/to/anvil-v0.24.2 \
  --bundle linux-x64=/path/to/brokk-anvil-v0.24.2-x86_64-unknown-linux-gnu
```

The GitHub Actions publish workflow is manual and defaults to build-only. Do
not enable publishing until all six package names have been bootstrapped under
the `@brokkai` organization and each has `publish-npm.yml` configured as its
npm trusted publisher.
