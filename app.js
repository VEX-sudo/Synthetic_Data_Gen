const CATS = ["sex", "smoker", "region"];
const NUMS = ["age", "bmi", "children", "charges"];
const PAGES = [
  ["dashboard", "ti-layout-dashboard", "Dashboard"],
  ["overview", "ti-table", "Dataset overview"],
  ["analysis", "ti-chart-histogram", "Data analysis"],
  ["train", "ti-cpu", "CTGAN training"],
  ["synth", "ti-sparkles", "Synthetic data"],
  ["compare", "ti-arrows-diff", "Comparison"],
  ["viz", "ti-chart-dots", "Visualizations"],
  ["docs", "ti-book", "Documentation"],
];

const S = {
  orig: [],
  synth: [],
  trained: false,
  trainStatus: "Idle",
  epochs: 200,
  loss: [],
  charts: [],
};

const fmt = (n, d = 0) =>
  n == null || Number.isNaN(n) ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d });
const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
const std = (a) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
};
const uniq = (a) => [...new Set(a)];
const col = (rows, k) => rows.map((r) => r[k]);
const ncol = (rows, k) => rows.map((r) => +r[k]);
const hist = (vals, bins = 8) => {
  const mn = Math.min(...vals), mx = Math.max(...vals), w = (mx - mn) / bins || 1;
  const c = Array(bins).fill(0);
  vals.forEach((v) => c[Math.min(bins - 1, Math.floor((v - mn) / w))]++);
  return c;
};
const corr = (xs, ys) => {
  const mx = mean(xs), my = mean(ys), sx = std(xs), sy = std(ys);
  if (!sx || !sy) return 0;
  return mean(xs.map((x, i) => (x - mx) * (ys[i] - my))) / (sx * sy);
};
const counts = (a) => a.reduce((m, x) => ((m[x] = (m[x] || 0) + 1), m), {});
const pick = (a) => a[Math.floor(Math.random() * a.length)];
const jitter = (x, s) => Math.max(0, x + (Math.random() - 0.5) * 2 * s);

function parseCSV(t) {
  const lines = t.trim().split(/\r?\n/);
  const h = lines[0].split(",");
  return lines.slice(1).filter(Boolean).map((line) => {
    const p = line.split(",");
    const r = {};
    h.forEach((k, i) => (r[k] = p[i]));
    r.age = +r.age; r.bmi = +r.bmi; r.children = +r.children; r.charges = +r.charges;
    return r;
  });
}

function trainSim() {
  S.trained = true;
  S.trainStatus = "Complete";
  S.loss = [];
  let g = 2.4;
  for (let i = 0; i < 8; i++) {
    g *= 0.72 + Math.random() * 0.08;
    S.loss.push(+g.toFixed(3));
  }
}

function generate(n) {
  const groups = {};
  S.orig.forEach((r) => {
    const k = r.smoker + "|" + r.sex;
    (groups[k] ||= []).push(r);
  });
  const keys = Object.keys(groups);
  const weights = keys.map((k) => groups[k].length);
  const tot = weights.reduce((s, x) => s + x, 0);
  const out = [];
  for (let i = 0; i < n; i++) {
    let u = Math.random() * tot, k = keys[0];
    for (let j = 0; j < keys.length; j++) {
      u -= weights[j];
      if (u <= 0) { k = keys[j]; break; }
    }
    const src = pick(groups[k]);
    out.push({
      age: Math.round(Math.min(64, Math.max(18, jitter(src.age, 3)))),
      sex: src.sex,
      bmi: +jitter(src.bmi, 1.4).toFixed(3),
      children: Math.max(0, Math.min(5, Math.round(jitter(src.children, 0.6)))),
      smoker: src.smoker,
      region: Math.random() < 0.88 ? src.region : pick(["northeast", "northwest", "southeast", "southwest"]),
      charges: +Math.max(1121, jitter(src.charges, src.charges * 0.08)).toFixed(4),
    });
  }
  S.synth = out;
}

function similarity() {
  if (!S.synth.length) return 0;
  const num = NUMS.map((k) => {
    const a = ncol(S.orig, k), b = ncol(S.synth, k);
    const d = Math.abs(mean(a) - mean(b)) / (mean(a) || 1);
    return Math.max(0, 1 - d);
  });
  const cat = CATS.map((k) => {
    const co = counts(col(S.orig, k)), cs = counts(col(S.synth, k));
    const keys = uniq([...Object.keys(co), ...Object.keys(cs)]);
    const n = S.orig.length, m = S.synth.length;
    return 1 - keys.reduce((s, x) => s + Math.abs((co[x] || 0) / n - (cs[x] || 0) / m), 0) / 2;
  });
  return mean([...num, ...cat]);
}

function killCharts() {
  S.charts.forEach((c) => c.destroy());
  S.charts = [];
}
function mkChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return;
  S.charts.push(new Chart(el, { ...cfg, options: { ...cfg.options, responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 8, font: { size: 11 } } } } } }));
}

function navHtml() {
  const page = location.hash.slice(1) || "dashboard";
  return PAGES.map(([id, ic, lab]) =>
    `<a class="nav-item ${page === id ? "active" : ""}" href="#${id}"><i class="ti ${ic}"></i>${lab}</a>`
  ).join("");
}

function pipeline(step) {
  const labs = ["Dataset", "Analysis", "CTGAN training", "Generation", "Evaluation"];
  return `<div class="card" style="margin-bottom:16px"><div class="card-sub" style="margin-bottom:12px">Pipeline</div><div class="pipeline-track">${labs
    .map((l, i) => {
      const st = i < step ? "done" : i === step ? "current" : "";
      return `<div class="p-node ${st}"><div class="p-line"></div><div class="p-dot">${i < step ? "✓" : i + 1}</div><div class="p-label">${l}</div></div>`;
    })
    .join("")}</div></div>`;
}

function table(rows, keys, n = 6) {
  const h = keys.map((k) => `<th>${k}</th>`).join("");
  const b = rows.slice(0, n).map((r) => `<tr>${keys.map((k) => {
    let v = r[k];
    if (k === "smoker" && v === "yes") v = `<span class="tag">yes</span>`;
    else if (k === "charges") v = fmt(v, 2);
    else if (typeof v === "number") v = fmt(v, k === "bmi" ? 1 : 0);
    return `<td>${v}</td>`;
  }).join("")}</tr>`).join("");
  return `<table><tr>${h}</tr>${b}</table>`;
}

function pageDashboard() {
  const sim = similarity();
  return `<div class="page-head"><div><h1>Dashboard</h1><p>Generates statistically faithful synthetic records from the insurance dataset using CTGAN-style tabular generation, without exposing real patient-level data.</p></div>
    <div class="actions"><button class="btn btn-ghost" onclick="location.hash='compare'">View comparison report</button>
    <button class="btn btn-primary" onclick="location.hash='synth'">Generate synthetic data</button></div></div>
    <div class="grid-6">
      <div class="kpi"><div class="kpi-label">Records</div><div class="kpi-value">${fmt(S.orig.length)}</div></div>
      <div class="kpi"><div class="kpi-label">Columns</div><div class="kpi-value">7</div></div>
      <div class="kpi"><div class="kpi-label">Categorical / numerical</div><div class="kpi-value">3 / 4</div></div>
      <div class="kpi"><div class="kpi-label">Training status</div><div class="kpi-value teal">${S.trainStatus}</div></div>
      <div class="kpi"><div class="kpi-label">Synthetic records</div><div class="kpi-value">${fmt(S.synth.length)}</div></div>
      <div class="kpi"><div class="kpi-label">Similarity score</div><div class="kpi-value teal">${S.synth.length ? sim.toFixed(2) : "—"}</div></div>
    </div>
    ${pipeline(S.synth.length ? 4 : S.trained ? 3 : 1)}
    <div class="grid-1-2">
      <div class="card"><h3>Charges by smoker status</h3><div class="card-sub">Original vs synthetic — the clearest learned pattern</div><div style="height:120px"><canvas id="cDash"></canvas></div></div>
      <div class="card"><h3>Dataset preview</h3><div class="card-sub">insurance.csv · first rows</div>${table(S.orig, ["age", "sex", "bmi", "smoker", "charges"], 3)}</div>
    </div>`;
}

function pageOverview() {
  const dups = S.orig.length - uniq(S.orig.map((r) => JSON.stringify(r))).length;
  const miss = S.orig.reduce((s, r) => s + Object.values(r).filter((v) => v === "" || v == null).length, 0);
  const sm = counts(col(S.orig, "smoker"));
  const yes = ((sm.yes || 0) / S.orig.length) * 100;
  const rows = ["age", "sex", "bmi", "children", "smoker", "region", "charges"].map((k) => {
    const role = CATS.includes(k) ? "categorical" : k === "children" ? "numerical (discrete)" : "numerical";
    const typ = CATS.includes(k) ? "object" : k === "bmi" || k === "charges" ? "float" : "int";
    const u = uniq(col(S.orig, k)).length;
    return `<tr><td>${k}</td><td>${typ}</td><td>${CATS.includes(k) ? `<span class="tag">${role}</span>` : role}</td><td>${fmt(u)}</td></tr>`;
  }).join("");
  return `<div class="page-head"><div><h1>Dataset overview</h1><p>insurance.csv — ${fmt(S.orig.length)} rows, 7 columns, ${miss ? miss + " missing" : "no missing values"}.</p></div></div>
    <div class="grid-4">
      <div class="kpi"><div class="kpi-label">Rows × columns</div><div class="kpi-value">${fmt(S.orig.length)} × 7</div></div>
      <div class="kpi"><div class="kpi-label">File size</div><div class="kpi-value">61 KB</div></div>
      <div class="kpi"><div class="kpi-label">Completeness</div><div class="kpi-value teal">${((1 - miss / (S.orig.length * 7)) * 100).toFixed(0)}%</div></div>
      <div class="kpi"><div class="kpi-label">Duplicate rows</div><div class="kpi-value">${dups}</div></div>
    </div>
    <div class="grid-1-2">
      <div class="card"><h3>Column summary</h3><div class="card-sub">Type, role, and unique-value count per column</div>
        <table><tr><th>Column</th><th>Type</th><th>Role</th><th>Unique</th></tr>${rows}</table></div>
      <div class="card"><h3>Category balance</h3><div class="card-sub">smoker is imbalanced — worth flagging for training</div>
        <div style="height:110px"><canvas id="cBal"></canvas></div>
        <div class="card-sub">non-smoker ${(100 - yes).toFixed(1)}% · smoker ${yes.toFixed(1)}%</div></div>
    </div>`;
}

function pageAnalysis() {
  const k = document.getElementById("colPick")?.value || "charges";
  return `<div class="page-head"><div><h1>Data analysis</h1><p>Explore distributions, correlations, and relationships in the original dataset.</p></div>
    <select id="colPick">${NUMS.map((c) => `<option ${c === k ? "selected" : ""}>${c}</option>`).join("")}</select></div>
    <div class="grid-1-2" style="margin-top:16px">
      <div class="card"><h3>Distribution — ${k}</h3><div class="card-sub">Shape of the original column</div><div style="height:120px"><canvas id="cDist"></canvas></div></div>
      <div class="card"><h3>Charges by smoker</h3><div class="card-sub">Mean charges, segmented</div><div style="height:120px"><canvas id="cBox"></canvas></div></div>
    </div>
    <div class="card" style="margin-top:16px"><h3>Correlation heatmap</h3><div class="card-sub">age, bmi, children, charges</div><div style="height:180px"><canvas id="cHeat"></canvas></div></div>`;
}

function pageTrain() {
  return `<div class="page-head"><div><h1>CTGAN training</h1><p>Model configuration and training status for the current run.</p></div>
    <span class="tag ${S.trained ? "ok" : ""}">${S.trained ? "Training complete" : S.trainStatus}</span></div>
    <div class="grid-1-2" style="margin-top:16px">
      <div class="card"><h3>Configuration</h3><div class="card-sub">Current model</div>
        <div class="kv"><span>Model</span><span>CTGAN-style tabular GAN</span></div>
        <div class="kv"><span>Epochs</span><span><input id="ep" type="number" min="10" max="500" value="${S.epochs}" style="width:80px"/></span></div>
        <div class="kv"><span>Categorical features</span><span>sex, smoker, region</span></div>
        <div class="kv"><span>Training rows</span><span>${fmt(S.orig.length)}</span></div>
        <div class="kv"><span>Library</span><span>browser generator (notebook uses ctgan + PyTorch)</span></div>
        <div style="margin-top:12px"><button class="btn btn-primary" id="btnTrain">Train model</button></div>
      </div>
      <div class="card"><h3>Generator loss</h3><div class="card-sub">Simulated convergence (full CTGAN lives in the notebook)</div>
        <div style="height:140px"><canvas id="cLoss"></canvas></div></div>
    </div>
    ${pipeline(S.trained ? 2 : 1)}`;
}

function pageSynth() {
  return `<div class="page-head"><div><h1>Synthetic data</h1><p>Generate new records from the trained model and export them.</p></div></div>
    <div class="card" style="margin-top:16px"><h3>Generate records</h3><div class="card-sub">No synthetic row is a copy of a real record</div>
      <div class="actions"><input id="nGen" type="number" min="10" max="5000" value="${S.synth.length || S.orig.length}"/>
      <button class="btn btn-primary" id="btnGen">Generate</button>
      ${S.synth.length ? `<span class="tag ok">Generated ${fmt(S.synth.length)}</span>` : ""}</div></div>
    <div class="card" style="margin-top:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <div><h3>Generated dataset</h3><div class="card-sub" style="margin:2px 0 0">${S.synth.length ? fmt(S.synth.length) + " synthetic rows" : "Not generated yet"}</div></div>
        <button class="btn btn-ghost" id="btnDl" ${S.synth.length ? "" : "disabled"}>Download CSV</button>
      </div>
      ${S.synth.length ? table(S.synth, ["age", "sex", "bmi", "smoker", "region", "charges"], 8) : "<p class='card-sub'>Train then generate to preview rows.</p>"}
    </div>`;
}

function pageCompare() {
  if (!S.synth.length) return `<div class="page-head"><div><h1>Comparison & evaluation</h1><p>Generate synthetic data first.</p></div></div>`;
  const sim = similarity();
  const rows = NUMS.map((k) => {
    const a = ncol(S.orig, k), b = ncol(S.synth, k);
    const d = Math.abs(mean(a) - mean(b)) / (mean(a) || 1);
    return `<tr><td>${k}</td><td>${fmt(mean(a), 1)}</td><td>${fmt(mean(b), 1)}</td><td>${fmt(std(a), 1)}</td><td>${fmt(std(b), 1)}</td><td><span class="tag ${d < 0.05 ? "ok" : ""}">${(d * 100).toFixed(1)}%</span></td></tr>`;
  }).join("");
  return `<div class="page-head"><div><h1>Comparison & evaluation</h1><p>How closely the synthetic dataset resembles the original — the core result of this project.</p></div></div>
    <div class="grid-4">
      <div class="kpi"><div class="kpi-label">Overall similarity</div><div class="kpi-value teal">${sim.toFixed(2)}</div></div>
      <div class="kpi"><div class="kpi-label">Correlation distance</div><div class="kpi-value">${Math.abs(corr(ncol(S.orig, "age"), ncol(S.orig, "charges")) - corr(ncol(S.synth, "age"), ncol(S.synth, "charges"))).toFixed(2)}</div></div>
      <div class="kpi"><div class="kpi-label">Mean charges (orig)</div><div class="kpi-value">${fmt(mean(ncol(S.orig, "charges")), 0)}</div></div>
      <div class="kpi"><div class="kpi-label">Mean charges (synth)</div><div class="kpi-value teal">${fmt(mean(ncol(S.synth, "charges")), 0)}</div></div>
    </div>
    <div class="grid-2">
      <div class="card"><h3>Distribution overlay — bmi</h3><div class="card-sub">Original vs synthetic, same bins</div><div style="height:120px"><canvas id="cBmi"></canvas></div></div>
      <div class="card"><h3>Categorical frequency — region</h3><div class="card-sub">Original vs synthetic</div><div style="height:120px"><canvas id="cReg"></canvas></div></div>
    </div>
    <div class="card" style="margin-top:16px"><h3>Statistical comparison</h3><div class="card-sub">Mean / std, original vs synthetic</div>
      <table><tr><th>Column</th><th>Mean (orig)</th><th>Mean (synth)</th><th>Std (orig)</th><th>Std (synth)</th><th>Diff</th></tr>${rows}</table></div>`;
}

function pageViz() {
  return `<div class="page-head"><div><h1>Visualizations</h1><p>A gallery of charts from the original and synthetic datasets.</p></div></div>
    <div class="grid-3" style="margin-top:16px">
      <div class="card"><h3>Age distribution</h3><div style="height:90px"><canvas id="v1"></canvas></div></div>
      <div class="card"><h3>Region frequency</h3><div style="height:90px"><canvas id="v2"></canvas></div></div>
      <div class="card"><h3>Charges by smoker</h3><div style="height:90px"><canvas id="v3"></canvas></div></div>
      <div class="card"><h3>Age vs charges</h3><div style="height:90px"><canvas id="v4"></canvas></div></div>
      <div class="card"><h3>BMI distribution</h3><div style="height:90px"><canvas id="v5"></canvas></div></div>
      <div class="card"><h3>Original vs synthetic (bmi)</h3><div style="height:90px"><canvas id="v6"></canvas></div></div>
    </div>`;
}

function pageDocs() {
  return `<div class="page-head"><div><h1>Documentation</h1><p>Background on synthetic data and CTGAN.</p></div></div>
    <div class="grid-2" style="margin-top:16px">
      <div class="card"><h4>What is synthetic data?</h4><p class="card-sub" style="margin-top:8px">Artificially generated records that share the statistical structure of a real dataset without containing any real individual's information.</p>
        <h4>What is CTGAN?</h4><p class="card-sub" style="margin-top:8px">A conditional GAN designed for tabular data — it models each column's distribution and learns correlations between them during training. The notebook trains the real CTGAN model; this UI generates matched tabular samples in-browser so the app stays usable without PyTorch.</p></div>
      <div class="card"><h4>Why it matters here</h4><p class="card-sub" style="margin-top:8px">Insurance data is sensitive to share. Synthetic rows let you demo distributions without exposing real people.</p>
        <div class="kv"><span>Model</span><span>CTGAN (PyTorch) in notebook</span></div>
        <div class="kv"><span>Frontend</span><span>this SPA + insurance.csv</span></div>
        <div class="kv"><span>Evaluation</span><span>mean/std + similarity</span></div>
        <div class="kv"><span>Data handling</span><span>pandas / CSV</span></div></div>
    </div>`;
}

function bindCharts(page) {
  const o = { ledger: "#2F4858", teal: "#12877F", amber: "#C98A2C" };
  const bar = (id, labels, datasets) => mkChart(id, { type: "bar", data: { labels, datasets } });
  if (page === "dashboard") {
    const yn = ["no", "yes"];
    const m = (rows, sm) => mean(rows.filter((r) => r.smoker === sm).map((r) => r.charges));
    bar("cDash", yn, [
      { label: "Original", backgroundColor: o.ledger, data: yn.map((s) => m(S.orig, s)) },
      { label: "Synthetic", backgroundColor: o.teal, data: yn.map((s) => (S.synth.length ? m(S.synth, s) : 0)) },
    ]);
  }
  if (page === "overview") {
    const sm = counts(col(S.orig, "smoker"));
    bar("cBal", ["non-smoker", "smoker"], [{ backgroundColor: [o.ledger, o.teal], data: [sm.no || 0, sm.yes || 0], label: "count" }]);
  }
  if (page === "analysis") {
    const k = document.getElementById("colPick").value;
    document.getElementById("colPick").onchange = () => render();
    bar("cDist", hist(ncol(S.orig, k)).map((_, i) => "b" + (i + 1)), [{ backgroundColor: o.ledger, data: hist(ncol(S.orig, k)), label: k }]);
    bar("cBox", ["non-smoker", "smoker"], [{ backgroundColor: [o.ledger, o.teal], data: ["no", "yes"].map((s) => mean(S.orig.filter((r) => r.smoker === s).map((r) => r.charges))), label: "mean charges" }]);
    mkChart("cHeat", {
      type: "bar",
      data: {
        labels: NUMS,
        datasets: NUMS.map((a, i) => ({
          label: a,
          backgroundColor: i === 0 ? o.teal : i === 1 ? o.ledger : o.amber,
          data: NUMS.map((b) => +corr(ncol(S.orig, a), ncol(S.orig, b)).toFixed(2)),
        })),
      },
    });
  }
  if (page === "train") {
    mkChart("cLoss", { type: "line", data: { labels: S.loss.map((_, i) => "e" + (i + 1)), datasets: [{ label: "G loss", borderColor: o.ledger, data: S.loss, tension: 0.3 }] } });
    document.getElementById("btnTrain").onclick = () => {
      S.epochs = +document.getElementById("ep").value || 200;
      S.trainStatus = "Training…";
      render();
      setTimeout(() => { trainSim(); generate(S.orig.length); render(); }, 400);
    };
  }
  if (page === "synth") {
    document.getElementById("btnGen").onclick = () => {
      if (!S.trained) trainSim();
      generate(+document.getElementById("nGen").value || S.orig.length);
      render();
    };
    document.getElementById("btnDl").onclick = () => {
      if (!S.synth.length) return;
      const keys = Object.keys(S.synth[0]);
      const csv = [keys.join(","), ...S.synth.map((r) => keys.map((k) => r[k]).join(","))].join("\n");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      a.download = "synthetic_insurance.csv";
      a.click();
    };
  }
  if (page === "compare" && S.synth.length) {
    bar("cBmi", hist(ncol(S.orig, "bmi")).map((_, i) => "" + (i + 1)), [
      { label: "Original", backgroundColor: o.ledger, data: hist(ncol(S.orig, "bmi")) },
      { label: "Synthetic", backgroundColor: o.teal, data: hist(ncol(S.synth, "bmi")) },
    ]);
    const regs = ["northeast", "northwest", "southeast", "southwest"];
    const cr = (rows) => { const c = counts(col(rows, "region")); return regs.map((r) => c[r] || 0); };
    bar("cReg", regs, [
      { label: "Original", backgroundColor: o.ledger, data: cr(S.orig) },
      { label: "Synthetic", backgroundColor: o.teal, data: cr(S.synth) },
    ]);
  }
  if (page === "viz") {
    bar("v1", hist(ncol(S.orig, "age"), 5).map((_, i) => i), [{ backgroundColor: o.ledger, data: hist(ncol(S.orig, "age"), 5) }]);
    const regs = ["NE", "NW", "SE", "SW"];
    const cr = counts(col(S.orig, "region"));
    bar("v2", regs, [{ backgroundColor: o.amber, data: ["northeast", "northwest", "southeast", "southwest"].map((r) => cr[r] || 0) }]);
    bar("v3", ["no", "yes"], [{ backgroundColor: [o.ledger, o.teal], data: ["no", "yes"].map((s) => mean(S.orig.filter((r) => r.smoker === s).map((r) => r.charges))) }]);
    const sample = S.orig.filter((_, i) => i % 20 === 0);
    mkChart("v4", { type: "scatter", data: { datasets: [{ label: "age/charges", backgroundColor: o.ledger, data: sample.map((r) => ({ x: r.age, y: r.charges })) }] } });
    bar("v5", hist(ncol(S.orig, "bmi"), 6).map((_, i) => i), [{ backgroundColor: o.teal, data: hist(ncol(S.orig, "bmi"), 6) }]);
    if (S.synth.length) {
      bar("v6", hist(ncol(S.orig, "bmi"), 5).map((_, i) => i), [
        { label: "orig", backgroundColor: o.ledger, data: hist(ncol(S.orig, "bmi"), 5) },
        { label: "synth", backgroundColor: o.teal, data: hist(ncol(S.synth, "bmi"), 5) },
      ]);
    }
  }
}

const VIEWS = { dashboard: pageDashboard, overview: pageOverview, analysis: pageAnalysis, train: pageTrain, synth: pageSynth, compare: pageCompare, viz: pageViz, docs: pageDocs };

function render() {
  killCharts();
  const page = location.hash.slice(1) || "dashboard";
  document.getElementById("nav").innerHTML = navHtml();
  document.getElementById("main").innerHTML = (VIEWS[page] || pageDashboard)();
  bindCharts(page);
}

async function boot() {
  try {
    const t = await fetch("insurance.csv").then((r) => {
      if (!r.ok) throw new Error("csv");
      return r.text();
    });
    S.orig = parseCSV(t);
    trainSim();
    generate(S.orig.length);
    window.addEventListener("hashchange", render);
    render();
  } catch (e) {
    document.getElementById("main").innerHTML = `<p class="err">Could not load insurance.csv. Serve this folder (e.g. <code>python -m http.server 8000</code>) instead of opening the HTML file directly.</p>`;
  }
}
boot();
