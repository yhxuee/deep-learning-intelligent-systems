// Every claim labelled **[E]** (experimental evidence) must say whose evidence it is: a cited publication,
// or a named experiment of this book (a worked example, the spine project, a lab, a figure script, an
// experiment). A paragraph with an [E] label and none of these fails. (Writing standard, "Claims carry
// their evidence"; scientific audit of v2.0, recommendation 4.)
// Usage: node scripts/check-evidence.mjs [root ...]
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const roots = process.argv.length > 2 ? process.argv.slice(2) : ['src/content/docs/core', 'src/content/docs/zh-tw/core', 'src/content/docs/appendix', 'src/content/docs/zh-tw/appendix'];
const SOURCE = /\]\([^)]*references\/#|Worked Example|詳解例題|例題|[Ss]pine|主幹|Stage \d|階段 \d|\blab\b|實驗室|figures\/|Experiment \d|實驗 \d|§\d/;
// A legend mentions all four labels but makes no evidence claim of its own.
// Match it by content rather than by one exact introductory phrase.
const LEGEND = /(?:label|標記|標籤|標示).*?\*\*\[D\]\*\*.*?\*\*\[E\]\*\*.*?\*\*\[O\]\*\*.*?\*\*\[H\]\*\*/is;
let bad = 0, labels = 0;
for (const root of roots) {
  (function walk(d) {
    for (const f of readdirSync(d)) {
      const p = join(d, f);
      if (statSync(p).isDirectory()) { walk(p); continue; }
      if (!/\.mdx?$/.test(p)) continue;
      const paras = readFileSync(p, 'utf8').replace(/^---[\s\S]*?\n---\n/, '').split(/\n\s*\n/);
      for (const para of paras) {
        if (!para.includes("**[E]**") || LEGEND.test(para.trim())) continue;
        labels++;
        if (!SOURCE.test(para)) { bad++; console.log(`${p}: [E] without a source — ${para.replace(/\s+/g, ' ').slice(0, 140)}`); }
      }
    }
  })(root);
}
console.log(`${labels} paragraphs with [E] labels; ${bad} without a named source`);
process.exit(bad ? 1 : 0);
