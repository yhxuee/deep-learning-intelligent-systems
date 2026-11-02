// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkBaseLinks from './src/plugins/remark-base-links.mjs';

// GitHub Pages project sites live under /<repo>/. Override with SITE / BASE env vars
// (the deploy workflow sets them from actions/configure-pages).
const SITE = process.env.SITE ?? 'https://yhxuee.github.io';
const BASE = process.env.BASE ?? '/deep-learning-intelligent-systems';
const REPO = process.env.REPO_URL ?? 'https://github.com/yhxuee/deep-learning-intelligent-systems';

export default defineConfig({
  site: SITE,
  base: BASE,
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, [remarkBaseLinks, { base: BASE }]],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [
    starlight({
      title: 'Deep Learning and Intelligent Systems',
      description:
        'An AI-native deep learning textbook: a 12-unit Core, a verification-first Atlas, portable AI Mentor prompts, and runnable Labs.',
      social: [{ icon: 'github', label: 'GitHub', href: REPO }],
      editLink: { baseUrl: `${REPO}/edit/main/` },
      lastUpdated: true,
      components: { MarkdownContent: './src/components/MarkdownContent.astro' },
      customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
      head: [
        { tag: 'link', attrs: { rel: 'alternate', type: 'text/plain', title: 'llms.txt', href: `${BASE}/llms.txt` } },
      ],
      sidebar: [
        { label: 'Start here', items: [
          { label: 'Overview', link: '/' },
          { slug: 'start/philosophy' },
          { slug: 'start/how-to-study' },
          { slug: 'start/changelog' },
        ] },
        { label: 'The Core', items: [
          { slug: 'core' },
          { label: 'I · The Nature of Learning', items: ['core/unit-01', 'core/unit-02', { label: '03 · Optimization', collapsed: true, items: [{ autogenerate: { directory: 'core/unit-03' } }] }] },
          { label: 'II · Representation, Memory, Computation', items: [{ label: '04 · Interaction', collapsed: true, items: [{ autogenerate: { directory: 'core/unit-04' } }] }, { label: '05 · Memory', collapsed: true, items: [{ autogenerate: { directory: 'core/unit-05' } }] }, 'core/unit-06'] },
          { label: 'III · Prediction, Generation, Decision', items: ['core/unit-07', 'core/unit-08', 'core/unit-09'] },
          { label: 'IV · Evidence, Reliability, Discovery', items: ['core/unit-10', 'core/unit-11', 'core/unit-12'] },
        ] },
        { label: 'Spine project', items: [{ slug: 'spine' }] },
        { label: 'The Atlas', items: [{ slug: 'atlas' }, { label: 'Nodes', items: [{ autogenerate: { directory: 'atlas/nodes' } }] }] },
        { label: 'The Mentor', items: [
          { slug: 'mentor' },
          { slug: 'mentor/modes' },
          { slug: 'mentor/unit-prompts' },
          { slug: 'mentor/machine-readable' },
        ] },
        { label: 'Laboratory', items: [{ slug: 'labs' }] },
        { label: 'Assessment & pathways', items: [
          { slug: 'assessment' },
          { slug: 'pathways/undergraduate' },
          { slug: 'pathways/graduate' },
        ] },
        { label: 'Project', items: [
          { slug: 'project/roadmap' },
          { slug: 'project/writing-standard' },
          { slug: 'project/governance' },
          { label: 'References', link: '/references/' },
        ] },
      ],
    }),
  ],
});
