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
      title: { en: 'Deep Learning and Intelligent Systems', 'zh-TW': '深度學習與智慧系統' },
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        'zh-tw': { label: '繁體中文（臺灣）', lang: 'zh-TW' },
      },
      description:
        'An AI-native deep learning textbook: a 12-unit Core, a verification-first Atlas, portable AI Mentor prompts, and runnable Labs.',
      social: [{ icon: 'github', label: 'GitHub', href: REPO }],
      editLink: { baseUrl: `${REPO}/edit/main/` },
      lastUpdated: true,
      components: { MarkdownContent: './src/components/MarkdownContent.astro' },
      customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
      head: [
        { tag: 'link', attrs: { rel: 'alternate', type: 'text/plain', title: 'llms.txt', href: `${BASE}/llms.txt` } },
        { tag: 'script', attrs: { type: 'module', src: `${BASE}/widgets/dlis-widgets.js` } },
      ],
      sidebar: [
        { label: 'Start here', translations: { 'zh-TW': '從這裡開始' }, items: [
          { slug: 'start/philosophy' },
          { slug: 'start/how-to-study' },
          { slug: 'start/glossary' },
          { slug: 'start/changelog' },
        ] },
        { label: 'The Core', translations: { 'zh-TW': '核心教材' }, items: [
          { slug: 'core' },
          { label: 'I · The Nature of Learning', translations: { 'zh-TW': '第一部 · 學習的本質' }, items: ['core/unit-01', 'core/unit-02', { label: '03 · Optimization', translations: { 'zh-TW': '03 · 最佳化' }, collapsed: true, items: [{ autogenerate: { directory: 'core/unit-03' } }] }] },
          { label: 'II · Representation, Memory, Computation', translations: { 'zh-TW': '第二部 · 表示、記憶與計算' }, items: [{ label: '04 · Interaction', translations: { 'zh-TW': '04 · 交互作用' }, collapsed: true, items: [{ autogenerate: { directory: 'core/unit-04' } }] }, { label: '05 · Memory', translations: { 'zh-TW': '05 · 記憶' }, collapsed: true, items: [{ autogenerate: { directory: 'core/unit-05' } }] }, 'core/unit-06'] },
          { label: 'III · Prediction, Generation, Decision', translations: { 'zh-TW': '第三部 · 預測、生成與決策' }, items: ['core/unit-07', 'core/unit-08', 'core/unit-09'] },
          { label: 'IV · Evidence, Reliability, Discovery', translations: { 'zh-TW': '第四部 · 證據、可靠性與發現' }, items: ['core/unit-10', 'core/unit-11', 'core/unit-12'] },
        ] },
        { label: 'Spine project', translations: { 'zh-TW': '主幹專案' }, items: [{ slug: 'spine' }] },
        { label: 'The Atlas', translations: { 'zh-TW': '知識圖譜 Atlas' }, items: [{ slug: 'atlas' }, { label: 'Nodes', translations: { 'zh-TW': '節點' }, items: [{ autogenerate: { directory: 'atlas/nodes' } }] }] },
        { label: 'The Mentor', translations: { 'zh-TW': 'AI 導師（Mentor）' }, items: [
          { slug: 'mentor' },
          { slug: 'mentor/modes' },
          { slug: 'mentor/unit-prompts' },
          { slug: 'mentor/machine-readable' },
        ] },
        { label: 'Laboratory', translations: { 'zh-TW': '實驗室' }, items: [{ slug: 'labs' }, { slug: 'labs/interactive' }] },
        { label: 'Assessment & pathways', translations: { 'zh-TW': '評量與學習路徑' }, items: [
          { slug: 'assessment' },
          { slug: 'pathways/undergraduate' },
          { slug: 'pathways/graduate' },
        ] },
        { label: 'Project', translations: { 'zh-TW': '專案' }, items: [
          { slug: 'project/roadmap' },
          { slug: 'project/writing-standard' },
          { slug: 'project/governance' },
          { slug: 'references' },
        ] },
      ],
    }),
  ],
});
