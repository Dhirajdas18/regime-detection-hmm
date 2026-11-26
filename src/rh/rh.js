// Gaussian HMM in the browser, mirrored from rh/hmm.py (single, unjittered start), tested against
// it through Node.
const RH = (() => {
  const dens = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
  function forward(x, p) {
    const K = p.mu.length, T = x.length, alpha = new Array(T), c = new Float64Array(T);
    let ll = 0;
    for (let t = 0; t < T; t++) {
      const a = new Float64Array(K);
      for (let j = 0; j < K; j++) {
        let s = 0;
        if (t === 0) s = p.pi[j];
        else for (let i = 0; i < K; i++) s += alpha[t - 1][i] * p.A[i][j];
        a[j] = s * dens(x[t], p.mu[j], p.sd[j]);
      }
      c[t] = a.reduce((u, v) => u + v, 0);
      for (let j = 0; j < K; j++) a[j] /= c[t];
      alpha[t] = a; ll += Math.log(c[t]);
    }
    return { alpha, c, ll };
  }
  function smooth(x, p) {
    const K = p.mu.length, T = x.length, { alpha, c, ll } = forward(x, p);
    const beta = new Array(T); beta[T - 1] = new Float64Array(K).fill(1);
    const xi = Array.from({ length: K }, () => new Float64Array(K));
    for (let t = T - 2; t >= 0; t--) {
      const bb = new Float64Array(K);
      for (let j = 0; j < K; j++) bb[j] = dens(x[t + 1], p.mu[j], p.sd[j]) * beta[t + 1][j];
      const b = new Float64Array(K);
      for (let i = 0; i < K; i++) {
        let s = 0;
        for (let j = 0; j < K; j++) { s += p.A[i][j] * bb[j]; xi[i][j] += p.A[i][j] * alpha[t][i] * bb[j] / c[t + 1]; }
        b[i] = s / c[t + 1];
      }
      beta[t] = b;
    }
    const gamma = alpha.map((a, t) => { const g = a.map((v, k) => v * beta[t][k]); const s = g.reduce((u, v) => u + v, 0); return g.map((v) => v / s); });
    return { gamma, xi, ll };
  }
  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function start(x, K) {
    const ab = x.map(Math.abs).sort((u, v) => u - v);
    const q = Array.from({ length: K + 1 }, (_, k) => quantile(ab, k / K));
    const mean = x.reduce((u, v) => u + v, 0) / x.length;
    const sd = Array.from({ length: K }, (_, k) => {
      const g = x.filter((v) => Math.abs(v) >= q[k] && Math.abs(v) <= q[k + 1]);
      const m = g.reduce((u, v) => u + v, 0) / g.length;
      return Math.max(Math.sqrt(g.reduce((u, v) => u + (v - m) ** 2, 0) / g.length), 1e-6);
    });
    const A = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => (i === j ? 0.95 : 0.05 / Math.max(K - 1, 1))));
    return { pi: new Array(K).fill(1 / K), A, mu: new Array(K).fill(mean), sd };
  }
  function fit(x, K, iters = 300, tol = 1e-7, sdFloor = 1e-4) {
    const p = start(x, K);
    let prev = -Infinity, ll = 0, it = 0;
    for (it = 0; it < iters; it++) {
      const s = smooth(x, p); ll = s.ll;
      const w = Array.from({ length: K }, (_, k) => s.gamma.reduce((u, g) => u + g[k], 0));
      p.pi = Array.from(s.gamma[0]);
      p.A = s.xi.map((r) => { const t = r.reduce((u, v) => u + v, 0); return Array.from(r, (v) => v / t); });
      p.mu = w.map((wk, k) => s.gamma.reduce((u, g, t) => u + g[k] * x[t], 0) / wk);
      p.sd = w.map((wk, k) => Math.max(Math.sqrt(s.gamma.reduce((u, g, t) => u + g[k] * (x[t] - p.mu[k]) ** 2, 0) / wk), sdFloor));
      if (ll - prev < tol * Math.abs(ll)) break;
      prev = ll;
    }
    const o = p.sd.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
    return { pi: o.map((i) => p.pi[i]), A: o.map((i) => o.map((j) => p.A[i][j])), mu: o.map((i) => p.mu[i]), sd: o.map((i) => p.sd[i]), ll, iters: it + 1 };
  }
  return { forward, smooth, fit };
})();
if (typeof module !== "undefined") module.exports = RH;
