import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
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
