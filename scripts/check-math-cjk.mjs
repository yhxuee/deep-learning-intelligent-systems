// Flag CJK characters or full-width punctuation inside $...$ / $$...$$ that are not wrapped in \text{...}.
// KaTeX renders such characters without font metrics; \text{} keeps them in the text font.
// Usage: node scripts/check-math-cjk.mjs <dir>
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
const root = process.argv[2] ?? 'src/content/docs/zh-tw';
const files = [];
(function walk(d) { for (const f of readdirSync(d)) { const p = join(d, f); statSync(p).isDirectory() ? walk(p) : /\.mdx?$/.test(p) && files.push(p); } })(root);
const CJK = /[　-〿㐀-鿿＀-￯]/;
function stripText(s) {
  // remove \text{...} with balanced braces
  let out = '', i = 0;
  while (i < s.length) {
    const m = s.slice(i).match(/^\\(text|textbf|mathrm|operatorname)\{/);
    if (m) { let depth = 1, j = i + m[0].length; while (j < s.length && depth) { if (s[j] === '{') depth++; else if (s[j] === '}') depth--; j++; } i = j; out += ' '; continue; }
    out += s[i++];
  }
  return out;
}
let bad = 0;
for (const f of files) {
  const src = readFileSync(f, 'utf8').replace(/^---[\s\S]*?\n---\n/, '').replace(/```[\s\S]*?```/g, '');
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
  let m;
  while ((m = re.exec(src))) {
    const body = m[1] ?? m[2];
    if (CJK.test(stripText(body))) { bad++; console.log(`${f}: ${body.slice(0, 90).replace(/\n/g, ' ')}`); }
  }
}
console.log(`${bad} math spans with bare CJK characters`);
process.exit(bad ? 1 : 0);
