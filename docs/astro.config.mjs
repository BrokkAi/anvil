import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import { defineConfig, fontProviders } from 'astro/config';
import rehypeBasePathLinks from './rehype-base-path-links.mjs';

const site = process.env.PUBLIC_DOCS_SITE ?? 'https://anvil.brokk.ai';
const productionBase = process.env.PUBLIC_DOCS_BASE ?? '/';
const isDev = process.argv.includes('dev');
const base = isDev ? '/' : productionBase;
const socialCardPath = [
  productionBase.replace(/^\/+|\/+$/g, ''),
  'anvil-social-card.png',
]
  .filter(Boolean)
  .join('/');
const socialCardUrl = new URL(`/${socialCardPath}`, site).href;

export default defineConfig({
  site,
  base,
  /*
   * Self-hosted and subset at build time. The Google Fonts <link> this replaces was a
   * render-blocking stylesheet on a third-party origin: two extra connections, then a
   * second round trip before any font file could even start downloading. Astro emits
   * the @font-face rules inline, preloads the files from our own origin, and derives
   * fallback metrics so swapping in the real face does not shift layout.
   *
   * Weights are exactly the ones the stylesheets ask for -- see --anvil-weight-* in
   * src/styles/anvil.css. Adding a weight here without a matching rule ships dead bytes.
   */
  fonts: [
    {
      name: 'Rajdhani',
      cssVariable: '--font-rajdhani',
      provider: fontProviders.google(),
      weights: [400, 600, 700],
      subsets: ['latin'],
      fallbacks: ['ui-sans-serif', 'system-ui', 'sans-serif'],
    },
    {
      name: 'JetBrains Mono',
      cssVariable: '--font-jetbrains-mono',
      provider: fontProviders.google(),
      weights: [400, 600, 700, 800],
      subsets: ['latin'],
      fallbacks: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
    },
    {
      name: 'Staatliches',
      cssVariable: '--font-staatliches',
      provider: fontProviders.google(),
      weights: [400],
      subsets: ['latin'],
      fallbacks: ['Impact', 'Haettenschweiler', 'Arial Narrow Bold', 'sans-serif'],
    },
  ],
  markdown: {
    processor: unified({
      rehypePlugins: [[rehypeBasePathLinks, { base }]],
    }),
  },
  integrations: [
    starlight({
      title: 'Anvil',
      description: 'A portable agent runtime for your ACP interface.',
      head: [
        { tag: 'meta', attrs: { property: 'og:image', content: socialCardUrl } },
        { tag: 'meta', attrs: { property: 'og:image:type', content: 'image/png' } },
        { tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
        { tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
        {
          tag: 'meta',
          attrs: {
            property: 'og:image:alt',
            content:
              'Anvil, the portable agent runtime for ACP clients: one agent engine, your interface.',
          },
        },
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
        { tag: 'meta', attrs: { name: 'twitter:image', content: socialCardUrl } },
        {
          tag: 'meta',
          attrs: {
            name: 'twitter:image:alt',
            content:
              'Anvil, the portable agent runtime for ACP clients: one agent engine, your interface.',
          },
        },
      ],
      customCss: ['./src/styles/anvil.css'],
      components: {
        Head: './src/components/AnvilHead.astro',
        Header: './src/components/AnvilHeader.astro',
        Hero: './src/components/AnvilHero.astro',
      },
      favicon: '/favicon.svg',
      editLink: {
        baseUrl: 'https://github.com/BrokkAi/anvil/edit/master/docs/',
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/BrokkAi/anvil',
        },
      ],
      sidebar: [
        {
          label: 'Start',
          items: [
            { label: 'Overview', slug: 'overview' },
            { label: 'Install Anvil', slug: 'install' },
            { label: '10-Minute Evaluation', slug: 'evaluate-anvil' },
            { label: 'License and Use Cases', slug: 'license-use-cases' },
            { label: 'Third-Party Notices', slug: 'third-party-notices' },
          ],
        },
        {
          label: 'Use Anvil',
          items: [
            { label: 'Zed', slug: 'zed' },
            { label: 'JetBrains', slug: 'jetbrains' },
            { label: 'Neovim', slug: 'neovim' },
            { label: 'Other ACP Clients', slug: 'other-acp-clients' },
            { label: 'Model Providers and Setup', slug: 'providers' },
            { label: 'Permissions and Sandboxing', slug: 'permissions-sandboxing' },
            { label: 'Tools and Managed Bifrost', slug: 'tools-bifrost' },
            { label: 'Sessions and Context', slug: 'sessions-context' },
          ],
        },
        {
          label: 'Extend Anvil',
          items: [
            { label: 'MCP Servers', slug: 'mcp' },
            { label: 'Skills and Plugins', slug: 'skills-plugins' },
            { label: 'Subagents', slug: 'subagents' },
            { label: 'Build an ACP Client', slug: 'build-acp-client' },
          ],
        },
        {
          label: 'Reference and Trust',
          items: [
            { label: 'Slash Commands', slug: 'slash-commands' },
            { label: 'Configuration and CLI', slug: 'configuration' },
            { label: 'Subagent Concurrency', slug: 'concurrency' },
            { label: 'Data and Trust Boundaries', slug: 'data-boundaries' },
          ],
        },
      ],
    }),
  ],
});
