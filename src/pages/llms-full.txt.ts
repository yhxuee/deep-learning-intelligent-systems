import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { url } from '../lib/atlas';

/** Concatenated Markdown source of the Core, Atlas and Mentor pages, for ingestion by AI tools. */
export const GET: APIRoute = async ({ site }) => {
  const docs = (await getCollection('docs'))
    .filter((d) => /^(start|core|spine|atlas|mentor)/.test(d.id))
    .sort((a, b) => a.id.localeCompare(b.id));
  const parts = docs.map((d) => `\n\n---\n# ${d.data.title}\nSource: ${new URL(url(d.id + '/'), site)}\n\n${d.body ?? ''}`);
  return new Response(`# Deep Learning and Intelligent Systems — full text${parts.join('')}\n`, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
};
