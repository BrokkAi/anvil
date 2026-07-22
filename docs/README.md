# Anvil documentation

The Anvil documentation site uses [Astro Starlight](https://starlight.astro.build/).

## Local development

```bash
cd docs
npm ci
npm run dev
```

Astro serves the site at the root during local development and production. The production site is deployed to the custom domain [`https://anvil.brokk.ai`](https://anvil.brokk.ai), so its default base path is `/`.

## Validation

```bash
npm run check
npm run check:evaluation
npm run build
```

The fixture check pins the documented evaluation symbols, paths, line numbers, and expected edit. The production build also checks that internal links and assets resolve under the configured deployment base. Documentation authors can continue to use root-relative links; the Rehype base-path plugin applies a non-root deployment base when `PUBLIC_DOCS_BASE` overrides the production default. Override the production URL when necessary with `PUBLIC_DOCS_SITE` and `PUBLIC_DOCS_BASE`.
