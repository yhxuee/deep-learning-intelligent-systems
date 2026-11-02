import { visit } from 'unist-util-visit';

/**
 * Prefix root-relative Markdown links ("/core/unit-03/") with the site base
 * ("/deep-learning-intelligent-systems"), so content can be written base-agnostic.
 */
export default function remarkBaseLinks({ base = '/' } = {}) {
  const prefix = base.replace(/\/$/, '');
  return (tree) => {
    if (!prefix) return;
    visit(tree, ['link', 'definition', 'image'], (node) => {
      const url = node.url;
      if (typeof url === 'string' && url.startsWith('/') && !url.startsWith('//') && !url.startsWith(prefix + '/')) {
        node.url = prefix + url;
      }
    });
  };
}
