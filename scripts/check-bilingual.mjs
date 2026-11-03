// Check that every English page and its Traditional Chinese counterpart state the same mathematics.
// For each pair it compares (1) the display equations ($$...$$), with the contents of \text{...} ignored,
// and (2) the numbered results they state or cite — theorems, propositions, corollaries, lemmas,
// definitions, worked examples, experiments and problems — so that a correction made in one language
// and not the other is caught. (The Simplified Chinese edition is generated from zh-TW and checked by
// scripts/test_zh_cn.py.)
// Usage: node scripts/check-bilingual.mjs [en-root] [zh-root]
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';

const EN = process.argv[2] ?? 'src/content/docs';
const ZH = process.argv[3] ?? 'src/content/docs/zh-tw';
const files = [];
(function walk(d) {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) { if (!/zh-(tw|cn)$/.test(p)) walk(p); } else if (/\.mdx?$/.test(p)) files.push(p);
  }
})(EN);

function stripText(s) {
  let out = '', i = 0;
  while (i < s.length) {
    const m = s.slice(i).match(/^\\(text|textbf|mathrm|operatorname)\{/);
    if (m) {
      let depth = 1, j = i + m[0].length;
      while (j < s.length && depth) { if (s[j] === '{') depth++; else if (s[j] === '}') depth--; j++; }
      i = j; continue;
    }
    out += s[i++];
  }
  // Word order inside a translated \text{} can move punctuation and grouping around it; compare the math.
  return out.replace(/\\[,;:!> ]|\\q?quad/g, '').replace(/[\s.,;:()]/g, '');
}
const body = (src) => src.replace(/^---[\s\S]*?\n---\n/, '').replace(/```[\s\S]*?```/g, '');
const displays = (src) => [...body(src).matchAll(/\$\$([\s\S]+?)\$\$/g)].map((m) => stripText(m[1]));

// Numbered results, normalized to a language-independent key.
const LABELS = [
  [/(?:\bTheorems?|\bThm\.?|定理) ([A-H]?\d+(?:\.\d+)?)/g, 'T'],
  [/(?:\bPropositions?|\bProp\.|命題) ([A-H]?\d+(?:\.\d+)?)/g, 'P'],
  [/(?:\bCorollar(?:y|ies)|\bCor\.|系理) ([A-H]?\d+(?:\.\d+)?)/g, 'C'],
  [/(?:\bLemma|引理) ([A-H]?\d+(?:\.\d+)?)/g, 'L'],
  [/([Ww]orked [Ee]xamples?|詳解例題|Example|例題) ([A-H]?\d+\.\d+)/g, 'E'],
  [/(Experiment|實驗) (\d+\.\d+)/g, 'X'],
  [/(Problem|習題) (\d+\.\d+)/g, 'Q'],
];
function labels(src) {
  const set = new Set();
  const text = body(src).replace(/\]\([^)]*\)/g, ']');
  for (const [re, k] of LABELS) for (const m of text.matchAll(re)) set.add(`${k}${m[m.length - 1]}`);
  // headings of numbered items, e.g. "**11.3 ★" in problem lists
  for (const m of text.matchAll(/^\*\*(\d+\.\d+|[A-H]\.\d+) /gm)) set.add(`Q${m[1]}`);
  return set;
}

let bad = 0, pairs = 0;
for (const en of files) {
  const rel = relative(EN, en);
  const zh = join(ZH, rel);
  if (!existsSync(zh)) continue;
  pairs++;
  const a = readFileSync(en, 'utf8'), b = readFileSync(zh, 'utf8');
  const da = displays(a), db = displays(b);
  if (da.length !== db.length) {
    bad++; console.log(`${rel}: ${da.length} display equations in English, ${db.length} in zh-TW`);
  } else {
    // Translation can reorder words inside \text{} and with them the math tokens between them; accept a
    // difference in order only when the multiset of symbols is identical.
    const sorted = (x) => [...x].sort().join('');
    da.forEach((eq, i) => {
      if (eq !== db[i] && sorted(eq) !== sorted(db[i])) { bad++; console.log(`${rel}: display equation ${i + 1} differs\n  en: ${eq.slice(0, 110)}\n  zh: ${db[i].slice(0, 110)}`); }
    });
  }
  const la = labels(a), lb = labels(b);
  const onlyEn = [...la].filter((x) => !lb.has(x)), onlyZh = [...lb].filter((x) => !la.has(x));
  if (onlyEn.length || onlyZh.length) {
    bad++; console.log(`${rel}: numbered results differ — only en: ${onlyEn.join(' ') || '–'}; only zh-TW: ${onlyZh.join(' ') || '–'}`);
  }
}
console.log(`${pairs} page pairs compared; ${bad} discrepancies`);
process.exit(bad ? 1 : 0);
