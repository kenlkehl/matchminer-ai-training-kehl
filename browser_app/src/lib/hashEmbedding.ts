export function hashTextEmbedding(text: string, dimensions = 384): number[] {
  const vector = new Array<number>(dimensions).fill(0);
  const tokens = text.toLowerCase().match(/[a-z0-9+-]+/g) ?? [];
  for (const token of tokens) {
    const idx = positiveHash(token) % dimensions;
    const sign = positiveHash(`sign:${token}`) % 2 === 0 ? 1 : -1;
    vector[idx] += sign * Math.log2(token.length + 1);
  }
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm ? vector.map((value) => value / norm) : vector;
}

function positiveHash(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
