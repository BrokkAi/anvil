import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import rehypeBasePathLinks from './rehype-base-path-links.mjs';

const site = process.env.PUBLIC_DOCS_SITE ?? 'https://brokkai.github.io';
const productionBase = process.env.PUBLIC_DOCS_BASE ?? '/anvil';
const isDev = process.argv.includes('dev');
const base = isDev ? '/' : productionBase;

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
      description: 'An ACP-native agent runtime for any interface.',
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
            { label: 'Quick start', slug: 'quick-start' },
          ],
        },
        {
          label: 'Build with Anvil',
          items: [
            { label: 'Core concepts', slug: 'concepts' },
            { label: 'ACP clients', slug: 'clients' },
            { label: 'Configuration', slug: 'configuration' },
          ],
        },
      ],
    }),
  ],
});
