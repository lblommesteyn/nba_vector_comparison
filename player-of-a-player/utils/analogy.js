export function analogy(A, B, C, embeddings) {
  const vec = (a, b, c) => a.map((_, i) => a[i] - b[i] + c[i]);

  const cosineSimilarity = (v1, v2) => {
    const dot = v1.reduce((sum, val, i) => sum + val * v2[i], 0);
    const mag1 = Math.sqrt(v1.reduce((sum, val) => sum + val * val, 0));
    const mag2 = Math.sqrt(v2.reduce((sum, val) => sum + val * val, 0));
    return dot / (mag1 * mag2);
  };

  const analogyVec = vec(embeddings[A], embeddings[B], embeddings[C]);

  let bestMatch = '';
  let bestScore = -Infinity;
  for (const [name, vector] of Object.entries(embeddings)) {
    if ([A, B, C].includes(name)) continue;
    const score = cosineSimilarity(analogyVec, vector);
    if (score > bestScore) {
      bestScore = score;
      bestMatch = name;
    }
  }

  return bestMatch;
}
