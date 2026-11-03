import { existsSync } from 'node:fs';
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

/** Sidebar entry for a Core unit: a collapsible ten-page group once the unit is a chapter folder. */
const UNIT_NAMES = {
  1: ['Learning', '學習', '学习'], 2: ['Representation', '表示', '表示'], 3: ['Optimization', '最佳化', '优化'],
  4: ['Interaction', '交互作用', '交互'], 5: ['Memory', '記憶', '记忆'], 6: ['Compute', '計算', '计算'],
  7: ['Self-supervision', '自監督', '自监督'], 8: ['Generation', '生成', '生成'], 9: ['Decision', '決策', '决策'],
  10: ['Failure', '失效', '失效'], 11: ['Interpretation', '詮釋', '解释'], 12: ['Discovery', '發現', '发现'],
};
function unitItem(n) {
  const id = String(n).padStart(2, '0');
  const [en, tw, cn] = UNIT_NAMES[n];
  if (!existsSync(new URL(`./src/content/docs/core/unit-${id}`, import.meta.url))) return `core/unit-${id}`;
  return { label: `${id} · ${en}`, translations: { 'zh-TW': `${id} · ${tw}`, 'zh-CN': `${id} · ${cn}` }, collapsed: true, items: [{ autogenerate: { directory: `core/unit-${id}` } }] };
}

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
      title: { en: 'Deep Learning and Intelligent Systems', 'zh-TW': '深度學習與智慧系統', 'zh-CN': '深度学习与智能系统' },
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        'zh-tw': { label: '繁體中文（臺灣）', lang: 'zh-TW' },
        'zh-cn': { label: '简体中文', lang: 'zh-CN' },
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
        { label: 'Start here', translations: { 'zh-TW': '從這裡開始', 'zh-CN': '从这里开始' }, items: [
          { slug: 'start/philosophy' },
          { slug: 'start/how-to-study' },
          { slug: 'start/notation' },
          { slug: 'start/glossary' },
          { slug: 'start/changelog' },
          { slug: 'start/errata' },
        ] },
        { label: 'The Core', translations: { 'zh-TW': '核心教材', 'zh-CN': '核心教材' }, items: [
          { slug: 'core' },
          { label: 'I · The Nature of Learning', translations: { 'zh-TW': '第一部 · 學習的本質', 'zh-CN': '第一部分 · 学习的本质' }, items: [1, 2, 3].map(unitItem) },
          { label: 'II · Representation, Memory, Computation', translations: { 'zh-TW': '第二部 · 表示、記憶與計算', 'zh-CN': '第二部分 · 表示、记忆与计算' }, items: [4, 5, 6].map(unitItem) },
          { label: 'III · Prediction, Generation, Decision', translations: { 'zh-TW': '第三部 · 預測、生成與決策', 'zh-CN': '第三部分 · 预测、生成与决策' }, items: [7, 8, 9].map(unitItem) },
          { label: 'IV · Evidence, Reliability, Discovery', translations: { 'zh-TW': '第四部 · 證據、可靠性與發現', 'zh-CN': '第四部分 · 证据、可靠性与发现' }, items: [10, 11, 12].map(unitItem) },
        ] },
        { label: 'Mathematical appendices', translations: { 'zh-TW': '數學附錄', 'zh-CN': '数学附录' }, items: [{ autogenerate: { directory: 'appendix' } }] },
        { label: 'Spine project', translations: { 'zh-TW': '主幹專案', 'zh-CN': '主干项目' }, items: [{ slug: 'spine' }] },
        { label: 'The Atlas', translations: { 'zh-TW': '知識圖譜 Atlas', 'zh-CN': '知识图谱 Atlas' }, items: [{ slug: 'atlas' }, { label: 'Nodes', translations: { 'zh-TW': '節點', 'zh-CN': '节点' }, items: [{ autogenerate: { directory: 'atlas/nodes' } }] }] },
        { label: 'The Mentor', translations: { 'zh-TW': 'AI 導師（Mentor）', 'zh-CN': 'AI 导师（Mentor）' }, items: [
          { slug: 'mentor' },
          { slug: 'mentor/modes' },
          { slug: 'mentor/unit-prompts' },
          { slug: 'mentor/machine-readable' },
        ] },
        { label: 'Laboratory', translations: { 'zh-TW': '實驗室', 'zh-CN': '实验室' }, items: [{ slug: 'labs' }, { slug: 'labs/interactive' }] },
        { label: 'Assessment & pathways', translations: { 'zh-TW': '評量與學習路徑', 'zh-CN': '评估与学习路径' }, items: [
          { slug: 'assessment' },
          { slug: 'pathways/undergraduate' },
          { slug: 'pathways/graduate' },
          { slug: 'pathways/reading-paths' },
        ] },
        { label: 'Project', translations: { 'zh-TW': '專案', 'zh-CN': '项目' }, items: [
          { slug: 'project/roadmap' },
          { slug: 'project/writing-standard' },
          { slug: 'project/governance' },
          { slug: 'references' },
        ] },
      ],
    }),
  ],
});
