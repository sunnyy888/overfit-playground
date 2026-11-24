// Tests for the overfitting core. Run: node test.mjs   (loads <script id="core"> from index.html)
import { readFileSync } from "node:fs";
const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const OF = new Function("module", html.match(/<script id="core">([\s\S]*?)<\/script>/)[1] + "; return OF;")({});
let pass = 0, fail = 0;
const check = (n, ok, d = "") => { ok ? pass++ : fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${n}${d ? ": " + d : ""}`); };
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

// 1. Legendre basis: P2(x) = (3x^2 - 1)/2, P3(x) = (5x^3 - 3x)/2, orthogonal on [-1, 1]
{
  const P = OF.legendre(0.3, 3);
  check("Legendre P2, P3 values", Math.abs(P[2] - (3 * 0.09 - 1) / 2) < 1e-12 && Math.abs(P[3] - (5 * 0.027 - 0.9) / 2) < 1e-12);
  let ip = 0; const N = 20000; for (let i = 0; i < N; i++) { const x = -1 + (2 * i + 1) / N, Q = OF.legendre(x, 5); ip += Q[2] * Q[5] * 2 / N; }
  check("Legendre P2 and P5 are orthogonal", Math.abs(ip) < 1e-6, ip.toExponential(1));
}
// 2. exact recovery: noiseless cubic, degree 3 fit -> zero error everywhere
{
  const f = x => 0.5 - x + 2 * x ** 3, rng = OF.mulberry32(1), d = OF.sample(rng, 20, f, 0), m = OF.fitOLS(d.xs, d.ys, 3);
  const grid = Array.from({ length: 50 }, (_, i) => -1 + i / 24.5), err = Math.max(...OF.predict(m, grid).map((p, i) => Math.abs(p - f(grid[i]))));
  check("noiseless cubic recovered exactly by degree 3", err < 1e-6, err.toExponential(1));
}
// 3. least squares = normal equations on a well-posed problem
{
  const rng = OF.mulberry32(2), d = OF.sample(rng, 40, OF.CURVES.wave, 0.3), X = OF.design(d.xs, 4), b = OF.lstsq(X, d.ys);
  const g = [0, 1, 2, 3, 4].map(j => X.reduce((s, r, i) => s + r[j] * (d.ys[i] - r.reduce((t, v, k) => t + v * b[k], 0)), 0));
  check("QR least squares satisfies the normal equations X'(y - Xb) = 0", Math.max(...g.map(Math.abs)) < 1e-9, Math.max(...g.map(Math.abs)).toExponential(1));
}
// 4. interpolation: n points, degree n-1 -> training RMSE ~ 0 (like HW4 4B degree 9 with 10 points)
{
  const rng = OF.mulberry32(3), d = OF.sample(rng, 10, OF.CURVES.wave, 0.3), m = OF.fitOLS(d.xs, d.ys, 9);
  check("degree 9 through 10 points: training RMSE ~ 0", OF.rmse(m, d.xs, d.ys) < 1e-4, OF.rmse(m, d.xs, d.ys).toExponential(1));
}
// 5. the bias-variance picture over 200 datasets: train falls monotonically, validation is U-shaped
{
  const R = 200, tr = Array(16).fill(0), va = Array(16).fill(0), val = OF.sample(OF.mulberry32(999), 2000, OF.CURVES.wave, 0.3);
  for (let r = 0; r < R; r++) {
    const d = OF.sample(OF.mulberry32(100 + r), 25, OF.CURVES.wave, 0.3);
    for (let k = 0; k <= 15; k++) { const m = OF.fitOLS(d.xs, d.ys, k); tr[k] += OF.rmse(m, d.xs, d.ys) / R; va[k] += Math.log(OF.rmse(m, val.xs, val.ys)) / R; }
  }
  const best = va.indexOf(Math.min(...va));
  check("average training RMSE never rises with degree", tr.every((v, k) => !k || v <= tr[k - 1] + 1e-12), tr.slice(0, 6).map(v => v.toFixed(3)).join(" ") + " …");
  check("average validation error is U-shaped (best in the middle)", best >= 3 && best <= 9 && va[0] > va[best] && va[15] > va[best], `best degree ${best}`);
}
// 6. Ridge: matches the closed form, shrinks monotonically
{
  const rng = OF.mulberry32(4), d = OF.sample(rng, 30, OF.CURVES.bump, 0.3), lam = 0.05, m = OF.fitRidge(d.xs, d.ys, 6, lam);
  const Z = m.st.apply(OF.design(d.xs, 6)), n = d.xs.length, p = 7;
  const g = [...Array(p).keys()].map(j => -(2 / n) * Z.reduce((s, r, i) => s + r[j] * (d.ys[i] - r.reduce((t, v, k) => t + v * m.beta[k], 0)), 0) + (j ? 2 * lam * m.beta[j] : 0));
  check("Ridge solution has zero gradient (closed form)", Math.max(...g.map(Math.abs)) < 1e-9, Math.max(...g.map(Math.abs)).toExponential(1));
  const norms = [1e-4, 1e-3, 1e-2, 1e-1, 1].map(l => Math.hypot(...OF.fitRidge(d.xs, d.ys, 10, l).beta.slice(1)));
  check("Ridge coefficient norm shrinks as λ grows", norms.every((v, i) => !i || v < norms[i - 1]), norms.map(v => v.toFixed(3)).join(" > "));
  check("Ridge keeps every coefficient non-zero", OF.fitRidge(d.xs, d.ys, 10, 0.1).beta.slice(1).every(b => Math.abs(b) > 1e-8));
}
// 7. LASSO: KKT conditions, exact zeros, sparsity grows with λ
{
  const rng = OF.mulberry32(5), d = OF.sample(rng, 30, OF.CURVES.wave, 0.3), lam = 0.02, m = OF.fitLasso(d.xs, d.ys, 12, lam);
  const Z = m.st.apply(OF.design(d.xs, 12)), n = d.xs.length, r = d.ys.map((y, i) => y - Z[i].reduce((s, z, j) => s + z * m.beta[j], 0));
  let kkt = true;
  for (let j = 1; j <= 12; j++) { const gj = Z.reduce((s, row, i) => s + row[j] * r[i], 0) / n; if (Math.abs(m.beta[j]) > 1e-12 ? Math.abs(gj - lam * Math.sign(m.beta[j])) > 1e-7 : Math.abs(gj) > lam + 1e-7) kkt = false; }
  check("LASSO solution satisfies the KKT conditions", kkt);
  const zeros = [1e-4, 1e-3, 1e-2, 0.05, 0.2].map(l => OF.fitLasso(d.xs, d.ys, 12, l).beta.slice(1).filter(b => b === 0).length);
  check("LASSO sets coefficients exactly to zero, more as λ grows", zeros[4] > zeros[0] && zeros.every((z, i) => !i || z >= zeros[i - 1] - 1) && zeros[4] >= 8, zeros.join(" ≤ "));
  const big = OF.fitLasso(d.xs, d.ys, 12, 10);
  check("huge λ: LASSO predicts the mean", big.beta.slice(1).every(b => b === 0) && Math.abs(big.beta[0] - mean(d.ys)) < 1e-12);
}
// 8. "new data" variance: high-degree fits swing far more between datasets than a moderate one
{
  const spread = deg => { const preds = []; for (let r = 0; r < 50; r++) { const d = OF.sample(OF.mulberry32(500 + r), 15, OF.CURVES.wave, 0.3); preds.push(OF.predict(OF.fitOLS(d.xs, d.ys, deg), [0.95])[0]); } const m = mean(preds); return Math.sqrt(mean(preds.map(p => (p - m) ** 2))); };
  const s3 = spread(3), s12 = spread(12);
  check("degree-12 predictions vary far more across datasets than degree 3", s12 > 3 * s3, `sd at x=0.95: degree 3 ${s3.toFixed(2)}, degree 12 ${s12.toFixed(2)}`);
}
console.log(`\n${pass}/${pass + fail} PASS`); process.exit(fail ? 1 : 0);
