import { getCollection, type CollectionEntry } from 'astro:content';

export type DocEntry = CollectionEntry<'docs'>;
export type RefEntry = CollectionEntry<'references'>;

const REPO = (import.meta.env.REPO_URL as string | undefined) ?? 'https://github.com/yhxuee/deep-learning-intelligent-systems';

export const LOCALES = ['zh-tw'] as const;
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
}

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
    for (const d of all) {
      for (const l of LOCALES) {
        if (d.id.startsWith(l + '/')) {
          if (!titles.has(l)) titles.set(l, new Map());
          titles.get(l)!.set(baseId(d.id), d.data.title);
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
      for (const t of d.data.unit!.atlas) if (!nodes.has(t)) errors.push(`unit ${n}.atlas → unknown node "${t}"`);
    }
    if (errors.length) throw new Error(`Atlas validation failed:\n  ${errors.join('\n  ')}`);
    return { nodes, units, refs, titles };
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
export function masteryLabel(m: string, locale?: string): string {
  return locale === 'zh-tw' ? MASTERY_LABEL_ZH[m] : MASTERY_LABEL[m];
}
export const MASTERY_CLASS: Record<string, string> = {
  internalize: 'tier tier-internalize',
  'understand-and-verify': 'tier tier-verify',
  'access-on-demand': 'tier tier-access',
};
