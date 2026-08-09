# PyPI publication

The official distribution name is `brokk-anvil`; the installed command is
`anvil`. The unscoped `anvil` name belongs to an unrelated PyPI project.

Tagged releases dispatch `.github/workflows/publish-pypi.yml` only after the
GitHub release, CI, docs, crates.io, and npm publication gates pass. The
workflow checks the Python version against `Cargo.toml` and the requested tag,
builds a wheel and sdist, runs the unit suite, then installs the wheel into a
clean environment and runs the matching checksum-verified native release.

## First publication

Before the first registry write, add a pending PyPI trusted publisher with:

- PyPI project name: `brokk-anvil`
- GitHub owner: `BrokkAi`
- GitHub repository: `anvil`
- Workflow: `publish-pypi.yml`
- Environment: `pypi-publish`

PyPI will create the project when the workflow first publishes. For the
bootstrap release, run the workflow from `master` with an existing matching
tag and `publish` enabled. Subsequent version tags dispatch it automatically.
No PyPI API token is stored in GitHub.
