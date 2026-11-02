import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { loadGraph, url } from '../lib/atlas';

/** Machine-readable Atlas: every node with typed links, references and raw Markdown body. */
export const GET: APIRoute = async ({ site }) => {
  const { nodes, units, refs } = await loadGraph();
  const docs = await getCollection('docs');
  const abs = (p: string) => new URL(url(p), site).toString();
  const sectionsOf = (unitId: string) =>
    docs
      .filter((d) => d.id.startsWith(unitId + '/'))
      .sort((a, b) => (a.data.sidebar.order ?? 0) - (b.data.sidebar.order ?? 0))
      .map((d) => ({ title: d.data.title, url: abs(d.id + '/'), markdown: d.body }));
  const body = {
    schema: 'dlis-atlas/1.1',
    generated: new Date().toISOString(),
    units: [...units.values()]
      .sort((a, b) => a.data.unit!.number - b.data.unit!.number)
      .map((u) => ({ ...u.data.unit, title: u.data.title, url: abs(u.id + '/'), markdown: u.body, sections: sectionsOf(u.id) })),
    nodes: [...nodes.values()].map((n) => ({
      ...n.data.atlas,
      last_reviewed: n.data.atlas!.last_reviewed.toISOString().slice(0, 10),
      title: n.data.title,
      description: n.data.description,
      url: abs(n.id + '/'),
      markdown: n.body,
    })),
    references: Object.fromEntries([...refs].map(([k, r]) => [k, r.data])),
  };
  return new Response(JSON.stringify(body, null, 2), { headers: { 'Content-Type': 'application/json' } });
};
