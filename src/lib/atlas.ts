import { getCollection, type CollectionEntry } from 'astro:content';

export type DocEntry = CollectionEntry<'docs'>;
export type RefEntry = CollectionEntry<'references'>;

const REPO = (import.meta.env.REPO_URL as string | undefined) ?? 'https://github.com/yhxuee/deep-learning-intelligent-systems';

export const LOCALES = ['zh-tw', 'zh-cn'] as const;
export type Locale = (typeof LOCALES)[number] | undefined;

/** Join the configured base path with a site-relative path, e.g. url('core/unit-03/'), optionally under a locale. */
export function url(path: string, locale?: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = path.replace(/^\//, '');
  return locale ? `${base}/${locale}/${clean}` : `${base}/${clean}`;
}

/** Strip a locale prefix from a docs entry id ("zh-tw/core/unit-03" -> "core/unit-03"). */
export function baseId(id: string): string {
  for (const l of LOCALES) if (id.startsWith(l + '/')) return id.slice(l.length + 1);
  return id;
}

export function isLocalized(id: string): boolean {
  return LOCALES.some((l) => id === l || id.startsWith(l + '/'));
}

export function repoUrl(path = ''): string {
  return path ? `${REPO}/tree/main/${path.replace(/^\//, '')}` : REPO;
}

let cache: Promise<Graph> | undefined;

export interface Graph {
  nodes: Map<string, DocEntry>;
  units: Map<number, DocEntry>;
  refs: Map<string, RefEntry>;
  /** locale -> (base id -> localized title) */
  titles: Map<string, Map<string, string>>;
  /** locale -> (unit number -> localized unit index page) */
  localUnits: Map<string, Map<number, DocEntry>>;
}

/** The unit index page in the given locale, falling back to English. */
export function unitIn(g: Graph, n: number, locale?: string): DocEntry {
  return (locale && g.localUnits.get(locale)?.get(n)) || g.units.get(n)!;
}

/** Unit that precedes / follows n in the book's argument (the chain closes: 12 → 1). */
export const prevUnit = (n: number) => (n === 1 ? 12 : n - 1);
export const nextUnit = (n: number) => (n === 12 ? 1 : n + 1);

/** Title of an English docs entry in the given locale, falling back to English. */
export function titleIn(g: Graph, entry: DocEntry, locale?: string): string {
  return (locale && g.titles.get(locale)?.get(entry.id)) || entry.data.title;
}

/**
 * Load the Atlas graph and validate every typed link. Any dangling reference
 * (unknown node id, unit number, or bibliography key) fails the build.
 */
export function loadGraph(): Promise<Graph> {
  cache ??= (async () => {
    const all = await getCollection('docs');
    const docs = all.filter((d) => !isLocalized(d.id));
    const titles = new Map<string, Map<string, string>>();
    const localUnits = new Map<string, Map<number, DocEntry>>();
    for (const d of all) {
      for (const l of LOCALES) {
        if (d.id.startsWith(l + '/')) {
          if (!titles.has(l)) titles.set(l, new Map());
          titles.get(l)!.set(baseId(d.id), d.data.title);
          if (d.data.unit) {
            if (!localUnits.has(l)) localUnits.set(l, new Map());
            localUnits.get(l)!.set(d.data.unit.number, d);
          }
        }
      }
    }
    const refs = new Map((await getCollection('references')).map((r) => [r.id, r]));
    const nodes = new Map<string, DocEntry>();
    const units = new Map<number, DocEntry>();
    for (const d of docs) {
      if (d.data.atlas) {
        if (nodes.has(d.data.atlas.id)) throw new Error(`Duplicate Atlas id ${d.data.atlas.id}`);
        nodes.set(d.data.atlas.id, d);
      }
      if (d.data.unit) units.set(d.data.unit.number, d);
    }
    const errors: string[] = [];
    for (const [id, d] of nodes) {
      const a = d.data.atlas!;
      for (const field of ['prerequisites', 'related', 'predecessors', 'alternatives'] as const) {
        for (const t of a[field]) if (!nodes.has(t)) errors.push(`${id}.${field} → unknown node "${t}"`);
      }
      for (const u of a.core_units) if (!units.has(u)) errors.push(`${id}.core_units → unknown unit ${u}`);
      for (const r of a.references) if (!refs.has(r)) errors.push(`${id}.references → unknown key "${r}"`);
    }
    for (const [n, d] of units) {
      const u = d.data.unit!;
      for (const t of u.atlas) if (!nodes.has(t)) errors.push(`unit ${n}.atlas → unknown node "${t}"`);
      for (const b of u.builds_on) if (b.unit >= n) errors.push(`unit ${n}.builds_on → unit ${b.unit} is not an earlier unit`);
    }
    // The argument must be a complete chain 1 → 2 → … → 12 (→ 1).
    for (let n = 1; n <= 12; n++) if (!units.has(n)) errors.push(`the argument of the book is missing unit ${n}`);
    // Localized unit pages must make the same structural claims as the English ones.
    for (const [l, m] of localUnits) {
      for (const [n, d] of m) {
        const en = units.get(n);
        if (!en) { errors.push(`${l} unit ${n} has no English counterpart`); continue; }
        const a = en.data.unit!, b = d.data.unit!;
        const same = (x: unknown, y: unknown) => JSON.stringify(x) === JSON.stringify(y);
        if (a.part !== b.part || a.status !== b.status || !same(a.atlas, b.atlas) || !same(a.builds_on.map((x) => x.unit), b.builds_on.map((x) => x.unit)))
          errors.push(`${l} unit ${n} disagrees with English on part/status/atlas/builds_on`);
      }
    }
    if (errors.length) throw new Error(`Atlas validation failed:\n  ${errors.join('\n  ')}`);
    return { nodes, units, refs, titles, localUnits };
  })();
  return cache;
}

export function shortCite(r: RefEntry): string {
  const first = r.data.authors.split(/,| and /)[0].trim();
  const etal = /,| and /.test(r.data.authors) ? ' et al.' : '';
  return `${first}${etal} (${r.data.year})`;
}

export function fullCite(r: RefEntry): string {
  return `${r.data.authors} (${r.data.year}). ${r.data.title}. ${r.data.venue}.`;
}

export const MASTERY_LABEL: Record<string, string> = {
  internalize: 'Internalize',
  'understand-and-verify': 'Understand & verify',
  'access-on-demand': 'Access on demand',
};
export const MASTERY_LABEL_ZH: Record<string, string> = {
  internalize: '內化（Internalize）',
  'understand-and-verify': '理解與驗證（Understand & verify）',
  'access-on-demand': '按需查閱（Access on demand）',
};
export const MASTERY_LABEL_CN: Record<string, string> = {
  internalize: '内化（Internalize）',
  'understand-and-verify': '理解与验证（Understand & verify）',
  'access-on-demand': '按需查阅（Access on demand）',
};
export function masteryLabel(m: string, locale?: string): string {
  return locale === 'zh-tw' ? MASTERY_LABEL_ZH[m] : locale === 'zh-cn' ? MASTERY_LABEL_CN[m] : MASTERY_LABEL[m];
}
export const MASTERY_CLASS: Record<string, string> = {
  internalize: 'tier tier-internalize',
  'understand-and-verify': 'tier tier-verify',
  'access-on-demand': 'tier tier-access',
};

/** Short names of the twelve units, used in diagrams. */
/** Short names of the twelve units, used in diagrams: [English, zh-TW, zh-CN]. */
export const UNIT_SHORT: Record<number, [string, string, string]> = {
  1: ['Learning', '學習', '学习'], 2: ['Representation', '表示', '表示'], 3: ['Optimization', '最佳化', '优化'],
  4: ['Interaction', '交互作用', '交互'], 5: ['Memory', '記憶', '记忆'], 6: ['Compute', '計算', '计算'],
  7: ['Self-supervision', '自監督', '自监督'], 8: ['Generation', '生成', '生成'], 9: ['Decision', '決策', '决策'],
  10: ['Failure', '失效', '失效'], 11: ['Interpretation', '詮釋', '解释'], 12: ['Discovery', '發現', '发现'],
};
export const PART_NAME: Record<number, [string, string, string]> = {
  1: ['Part I · The nature of learning', '第一部 · 學習的本質', '第一部分 · 学习的本质'],
  2: ['Part II · Representation, memory and computation', '第二部 · 表示、記憶與計算', '第二部分 · 表示、记忆与计算'],
  3: ['Part III · Prediction, generation and decision', '第三部 · 預測、生成與決策', '第三部分 · 预测、生成与决策'],
  4: ['Part IV · Evidence, reliability and discovery', '第四部 · 證據、可靠性與發現', '第四部分 · 证据、可靠性与发现'],
};
