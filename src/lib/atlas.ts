import { getCollection, type CollectionEntry } from 'astro:content';

export type DocEntry = CollectionEntry<'docs'>;
export type RefEntry = CollectionEntry<'references'>;

const REPO = (import.meta.env.REPO_URL as string | undefined) ?? 'https://github.com/yhxuee/deep-learning-intelligent-systems';

/** Join the configured base path with a site-relative path, e.g. url('core/unit-03/'). */
export function url(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = path.replace(/^\//, '');
  return `${base}/${clean}`;
}

export function repoUrl(path = ''): string {
  return path ? `${REPO}/tree/main/${path.replace(/^\//, '')}` : REPO;
}

let cache: Promise<Graph> | undefined;

export interface Graph {
  nodes: Map<string, DocEntry>;
  units: Map<number, DocEntry>;
  refs: Map<string, RefEntry>;
}

/**
 * Load the Atlas graph and validate every typed link. Any dangling reference
 * (unknown node id, unit number, or bibliography key) fails the build.
 */
export function loadGraph(): Promise<Graph> {
  cache ??= (async () => {
    const docs = await getCollection('docs');
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
    return { nodes, units, refs };
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
export const MASTERY_CLASS: Record<string, string> = {
  internalize: 'tier tier-internalize',
  'understand-and-verify': 'tier tier-verify',
  'access-on-demand': 'tier tier-access',
};
