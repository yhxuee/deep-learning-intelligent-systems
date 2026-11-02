// Verify that every internal link in the built site resolves to a file, and that
// every "#fragment" link into a page matches an element id on that page.
// Usage: node scripts/check-links.mjs [distDir] [base]
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';

const dist = process.argv[2] ?? 'dist';
const base = (process.argv[3] ?? process.env.BASE ?? '/deep-learning-intelligent-systems').replace(/\/$/, '');

const files = [];
(function walk(dir) {
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) walk(p);
    else if (p.endsWith('.html')) files.push(p);
  }
})(dist);

const idCache = new Map();
const idsOf = (file) => {
  if (!idCache.has(file)) {
    const html = readFileSync(file, 'utf8');
    idCache.set(file, new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
  }
  return idCache.get(file);
};

const resolveTarget = (path) => {
  const rel = path.slice(base.length).replace(/^\//, '');
  const candidates = [join(dist, rel), join(dist, rel, 'index.html')];
  return candidates.find((c) => existsSync(c) && statSync(c).isFile());
};

let errors = 0;
let checked = 0;
for (const file of files) {
  const html = readFileSync(file, 'utf8');
  for (const [, raw] of html.matchAll(/\shref="([^"]+)"/g)) {
    const href = raw.replace(/&amp;/g, '&');
    if (/^(https?:|mailto:|data:|javascript:)/.test(href)) continue;
    const [path, frag] = href.split('#');
    let target;
    if (path === '') target = file;
    else if (path.startsWith('/')) {
      if (!path.startsWith(base + '/') && path !== base) {
        console.error(`✗ ${relative(dist, file)} → ${href} (missing base ${base})`);
        errors++;
        continue;
      }
      target = resolveTarget(path);
    } else {
      // relative link: resolve against the page's URL directory
      const pageUrl = new URL('https://x' + base + '/' + relative(dist, file).replace(/index\.html$/, ''));
      target = resolveTarget(new URL(path, pageUrl).pathname);
    }
    checked++;
    if (!target) {
      console.error(`✗ ${relative(dist, file)} → ${href} (no such page)`);
      errors++;
    } else if (frag && target.endsWith('.html') && !idsOf(target).has(decodeURIComponent(frag)) && frag !== '_top') {
      console.error(`✗ ${relative(dist, file)} → ${href} (no #${frag} on target)`);
      errors++;
    }
  }
}
console.log(`${checked} internal links checked in ${files.length} pages; ${errors} broken.`);
process.exit(errors ? 1 : 0);
