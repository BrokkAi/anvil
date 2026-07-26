function normalizeBase(base) {
  if (!base || base === '/') return '/';
  return `/${base.replace(/^\/+|\/+$/g, '')}`;
}

function prefixBasePath(value, base) {
  if (
    typeof value !== 'string' ||
    base === '/' ||
    !value.startsWith('/') ||
    value.startsWith('//') ||
    value === base ||
    value.startsWith(`${base}/`)
  ) {
    return value;
  }
  return `${base}${value}`;
}

export default function rehypeBasePathLinks(options = {}) {
  const base = normalizeBase(options.base);

  return (tree) => {
    const stack = [tree];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node || typeof node !== 'object') continue;

      if (node.properties) {
        for (const property of ['href', 'src']) {
          node.properties[property] = prefixBasePath(node.properties[property], base);
        }
      }
      if (Array.isArray(node.children)) stack.push(...node.children);
    }
  };
}

export { normalizeBase, prefixBasePath };
