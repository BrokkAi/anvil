# Anvil documentation

The Anvil documentation site uses [Astro Starlight](https://starlight.astro.build/).

## Local development

```bash
cd docs
npm ci
npm run dev
```

Astro serves the site at the root during local development. Production builds use the `/anvil` base path by default.

## Validation

```bash
npm run check
npm run build
```

The production build also checks that internal links and assets resolve under the configured deployment base. Override the production URL when necessary with `PUBLIC_DOCS_SITE` and `PUBLIC_DOCS_BASE`.
