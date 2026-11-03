/**
 * Three editions: English (root), Traditional Chinese for Taiwan (zh-tw) and Simplified Chinese for the
 * mainland (zh-cn).  Components choose their fixed strings with pick(); everything else comes from
 * the localized content pages.
 */
export type Loc = string | undefined;
export const isZh = (l: Loc) => l === 'zh-tw' || l === 'zh-cn';
export const idx = (l: Loc): 0 | 1 | 2 => (l === 'zh-tw' ? 1 : l === 'zh-cn' ? 2 : 0);
export function pick<T>(l: Loc, en: T, tw: T, cn: T): T {
  return l === 'zh-tw' ? tw : l === 'zh-cn' ? cn : en;
}
