import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { file } from 'astro/loaders';

const mastery = z.enum(['internalize', 'understand-and-verify', 'access-on-demand']);

/** Metadata carried by Core unit pages. */
const unit = z.object({
  number: z.number().int().min(1).max(12),
  part: z.number().int().min(1).max(4),
  question: z.string(),
  anchor: z.string().describe('The one concrete exemplar every learner internalizes'),
  spine: z.string().describe('What this unit adds to the spine-project language model'),
  checkpoint: z.string().describe('The AI-off independent checkpoint'),
  status: z.enum(['complete', 'draft', 'outline']),
  atlas: z.array(z.string()).default([]),
});

/** Metadata carried by Atlas node pages (the verification layer). */
const atlas = z.object({
  id: z.string().regex(/^(math|mech|arch|method|opt|para)-[a-z0-9-]+$/),
  type: z.enum(['mathematics', 'mechanism', 'architecture', 'method', 'optimizer', 'paradigm']),
  mastery,
  core_units: z.array(z.number().int().min(1).max(12)).default([]),
  prerequisites: z.array(z.string()).default([]),
  related: z.array(z.string()).default([]),
  predecessors: z.array(z.string()).default([]),
  alternatives: z.array(z.string()).default([]),
  lab: z.string().optional(),
  references: z.array(z.string()).min(1),
  last_reviewed: z.coerce.date(),
});

const glossary = defineCollection({
  loader: file('src/data/glossary.yaml'),
  schema: z.object({
    en: z.string(),
    zh: z.string(),
    area: z.enum(['general', 'optimization', 'attention', 'memory', 'statistics', 'systems', 'book']),
    cn: z.string().optional(),
    note: z.string().optional(),
  }),
});

export const collections = {
  glossary,
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({ extend: z.object({ unit: unit.optional(), atlas: atlas.optional() }) }),
  }),
  references: defineCollection({
    loader: file('src/data/bibliography.yaml'),
    schema: z.object({
      authors: z.string(),
      year: z.number().int(),
      title: z.string(),
      venue: z.string(),
      url: z.string().url().optional(),
      kind: z.enum(['paper', 'book', 'course', 'software', 'standard', 'essay']).default('paper'),
    }),
  }),
};
