// Muon Teleportation site: theme, hero animation, and data-driven results.

    // Theme preference: saved locally; otherwise follow the operating-system preference.
    (() => {
      const root = document.documentElement;
      const button = document.getElementById("themeToggle");
      const meta = document.getElementById("theme-color-meta");
      const getPreferred = () => {
        try { return localStorage.getItem("muon-theme"); } catch (_) { return null; }
      };
      const applyTheme = (theme) => {
        root.dataset.theme = theme;
        const dark = theme === "dark";
        button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
        button.innerHTML = dark
          ? '<span class="theme-icon" aria-hidden="true">☀</span><span class="theme-label">Light</span>'
          : '<span class="theme-icon" aria-hidden="true">☾</span><span class="theme-label">Dark</span>';
        meta.setAttribute("content", dark ? "#0a0f1d" : "#fbfcfe");
        if (typeof updateChartTheme === "function") updateChartTheme(dark);
      };
      const saved = getPreferred();
      const systemDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      applyTheme(saved || (systemDark ? "dark" : "light"));
      button.addEventListener("click", () => {
        const next = root.dataset.theme === "dark" ? "light" : "dark";
        applyTheme(next);
        try { localStorage.setItem("muon-theme", next); } catch (_) {}
      });
    })();

// ---------------------------------------------------------------------------
// Hero animation
// ---------------------------------------------------------------------------
(() => {
  const svg = document.getElementById("heroSvg");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs = {}, parent = svg) => {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    parent.appendChild(n);
    return n;
  };
  const CX = 640, CY = 205, ANG = -14, RX = 64, RY = 21;
  const rad = ANG * Math.PI / 180;
  const pt = (k, t) => {
    const x = RX * k * Math.cos(t), y = RY * k * Math.sin(t);
    return [CX + x * Math.cos(rad) - y * Math.sin(rad), CY + x * Math.sin(rad) + y * Math.cos(rad)];
  };

  // loss contours
  for (let k = 7; k >= 1; k--) {
    el("ellipse", { class: "contour" + (k === 1 ? " min" : ""), cx: CX, cy: CY, rx: RX * k, ry: RY * k, transform: `rotate(${ANG} ${CX} ${CY})` });
  }

  // path of the optimizer
  const small = [pt(6.4, 3.55), pt(6.15, 3.5), pt(5.9, 3.45), pt(5.65, 3.4)];
  const tele = pt(5.65, 5.15);
  const big = [tele, pt(3.9, 5.0), pt(2.3, 4.7), pt(0.2, 0)];

  const steps = []; // elements revealed in order, with delays (ms)
  const addLine = (a, b, delay) => {
    const l = el("line", { class: "seg h", x1: a[0], y1: a[1], x2: b[0], y2: b[1] });
    steps.push([l, delay]);
  };
  const addDot = (p, cls, delay) => {
    const d = el("circle", { class: cls + " h", cx: p[0], cy: p[1], r: cls === "tdot" ? 6.5 : 4.5 });
    steps.push([d, delay]);
    return d;
  };
  const addLabel = (txt, x, y, cls, delay, anchor = "start") => {
    const t = el("text", { class: "lbl h" + (cls ? " " + cls : ""), x, y, "text-anchor": anchor });
    t.textContent = txt;
    steps.push([t, delay]);
  };

  addDot(small[0], "dot", 400);
  for (let i = 1; i < small.length; i++) { addLine(small[i - 1], small[i], 450); addDot(small[i], "dot", 0); }
  addLabel("small Muon steps", small[0][0] - 8, small[0][1] + 34, "", 200);

  // teleport hop (quadratic arc drawn outside the valley)
  const mx = (small[3][0] + tele[0]) / 2 - 150, my = (small[3][1] + tele[1]) / 2 - 10;
  const hop = el("path", { class: "hop", pathLength: 1, d: `M${small[3][0]} ${small[3][1]} Q${mx} ${my} ${tele[0]} ${tele[1]}` });
  const tdot = el("circle", { class: "tdot h", cx: tele[0], cy: tele[1], r: 7 });
  addLabel("teleport: same loss", mx - 38, my - 2, "t", 500, "end");

  for (let i = 1; i < big.length; i++) { addLine(big[i - 1], big[i], 520); addDot(big[i], "dot", 0); }
  addLabel("larger steps", big[1][0] + 4, big[1][1] - 18, "", 200, "middle");

  const all = [...svg.querySelectorAll(".h")];
  let timers = [];
  const clear = () => { timers.forEach(clearTimeout); timers = []; };
  const later = (fn, ms) => timers.push(setTimeout(fn, ms));
  const reset = () => { all.forEach(n => n.classList.remove("on")); hop.classList.remove("on"); tdot.classList.remove("on"); };
  const showAll = () => { all.forEach(n => n.classList.add("on")); hop.classList.add("on"); };

  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { showAll(); return; }

  const play = () => {
    clear(); reset();
    let t = 300;
    let hopDone = false;
    steps.forEach(([node, delay]) => {
      // insert the teleport hop right before the first "big" element after the hop label
      if (!hopDone && node.classList.contains("t")) {
        t += delay;
        later(() => { node.classList.add("on"); hop.classList.add("on"); }, t);
        t += 1000;
        later(() => tdot.classList.add("on"), t);
        hopDone = true;
        return;
      }
      t += delay;
      later(() => node.classList.add("on"), t);
    });
    later(play, t + 3500);
  };
  play();
})();

// ---------------------------------------------------------------------------
// Data-driven results
// ---------------------------------------------------------------------------
var trainingChart = null;
var updateChartTheme = (dark) => { if (trainingChart) applyChartColors(trainingChart, dark); };

function applyChartColors(chart, dark) {
  const text = dark ? "#a1adc2" : "#566174";
  const grid = dark ? "#232c42" : "#e1e5ef";
  const o = chart.options;
  o.scales.x.ticks.color = text; o.scales.y.ticks.color = text;
  o.scales.x.title.color = text; o.scales.y.title.color = text;
  o.scales.x.grid.color = grid; o.scales.y.grid.color = grid;
  o.plugins.legend.labels.color = dark ? "#e9eef9" : "#101828";
  chart.update("none");
}

(() => {
  const data = window.MUON_RESULTS;
  if (!data) { console.error("results data missing (data/results.js did not load)"); return; }
  const isDark = () => document.documentElement.dataset.theme === "dark";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // headline facts
  const facts = document.getElementById("facts");
  if (facts) facts.innerHTML = (data.facts || []).map(f => `<div><b>${esc(f.value)}</b><span>${esc(f.label)}</span></div>`).join("");

  // results table
  const tableHost = document.getElementById("resultsTable");
  if (tableHost) {
    let html = `<table class="results-table"><thead><tr><th>Positional encoding</th><th>Teleport</th><th>Accuracy (%)</th><th>Epoch speedup</th><th>Wall-clock speedup</th><th>Time / epoch</th></tr></thead><tbody>`;
    data.datasets.forEach(ds => {
      html += `<tr class="dataset-row"><th colspan="6" scope="colgroup">${esc(ds.name)}</th></tr>`;
      const seen = {};
      ds.rows.forEach(r => {
        const count = ds.rows.filter(x => x.encoding === r.encoding).length;
        const first = !seen[r.encoding]; seen[r.encoding] = true;
        const cell = (v) => `<td class="${r.best ? "best" : ""}">${esc(v)}</td>`;
        html += `<tr class="${first ? "pe-start" : ""}">` + (first ? `<th scope="rowgroup" rowspan="${count}">${esc(r.encoding)}</th>` : "")
          + `<td>${r.teleport ? "Yes" : "No"}</td>`
          + `<td class="${r.best ? "best" : ""}">${esc(r.accuracy)}</td>`
          + `<td>${esc(r.epochSpeedup)}</td><td>${esc(r.wallSpeedup)}</td><td>${esc(r.timePerEpoch)}</td></tr>`;
      });
    });
    tableHost.innerHTML = html + "</tbody></table>";
  }

  // training curves
  const canvas = document.getElementById("trainingChart");
  if (!canvas || typeof Chart === "undefined") { console.error("Chart.js did not load"); return; }
  let view = "epoch", active = 0;
  const picker = document.getElementById("datasetPicker");
  const nameEl = document.getElementById("activeDatasetName");
  const epochsEl = document.getElementById("activeDatasetEpochs");
  const noteEl = document.getElementById("chartNote");
  const epochCount = (ds) => ds.curves.epoch[0].mean.length;

  picker.innerHTML = data.datasets.map((ds, i) =>
    `<button type="button" class="dataset-option${i === 0 ? " is-selected" : ""}" data-i="${i}" aria-pressed="${i === 0}"><span class="dataset-option-name">${esc(ds.name)}</span><span class="dataset-option-meta">${epochCount(ds)} epochs</span></button>`).join("");

  const COLORS = { RoPE: { light: "#2f5fd0", dark: "#6f9bff" }, APE: { light: "#d9692f", dark: "#f0955f" } };
  const colorFor = (label) => (COLORS[label.split(",")[0].trim()] || COLORS.RoPE)[isDark() ? "dark" : "light"];
  const toPts = (arr) => arr.map(p => ({ x: p.x, y: p.y }));

  function build() {
    const ds = data.datasets[active];
    const series = ds.curves[view];
    const sets = [];
    series.forEach(s => {
      const c = colorFor(s.label);
      const dashed = /no teleport/i.test(s.label);
      sets.push({ label: s.label, data: toPts(s.mean), borderColor: c, backgroundColor: c, borderWidth: 2.4, borderDash: dashed ? [7, 5] : [], pointRadius: 0, pointHoverRadius: 4, tension: 0.25, _band: false });
    });
    series.forEach((s, i) => {
      const c = colorFor(s.label);
      const lowerIdx = sets.length;
      sets.push({ label: s.label + " (min)", data: toPts(s.lower), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false, tension: 0.25, _band: true });
      sets.push({ label: s.label + " (max)", data: toPts(s.upper), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, tension: 0.25, fill: lowerIdx, backgroundColor: c + "26", _band: true });
    });
    return sets;
  }

  function render() {
    const ds = data.datasets[active];
    nameEl.textContent = ds.name;
    epochsEl.textContent = epochCount(ds) + " epochs";
    canvas.setAttribute("aria-label", `${ds.name} validation accuracy versus ${view === "epoch" ? "epoch" : "wall-clock time"}`);
    const xTitle = view === "epoch" ? "Epoch" : "Wall-clock time (s)";
    if (!trainingChart) {
      trainingChart = new Chart(canvas, {
        type: "line",
        data: { datasets: build() },
        options: {
          responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
          interaction: { mode: "nearest", intersect: false, axis: "x" },
          parsing: false,
          scales: {
            x: { type: "linear", title: { display: true, text: xTitle }, ticks: { precision: 0 }, grid: {} },
            y: { title: { display: true, text: "Validation accuracy (%)" }, grid: {}, ticks: {} }
          },
          plugins: {
            legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, filter: (item, d) => !d.datasets[item.datasetIndex]._band } },
            tooltip: { filter: (item) => !item.dataset._band, callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(2)}%` } }
          }
        }
      });
    } else {
      trainingChart.data.datasets = build();
      trainingChart.options.scales.x.title.text = xTitle;
    }
    applyChartColors(trainingChart, isDark());
  }

  // re-colour series on theme change
  updateChartTheme = (dark) => {
    if (!trainingChart) return;
    trainingChart.data.datasets = build();
    applyChartColors(trainingChart, dark);
  };

  picker.addEventListener("click", (e) => {
    const b = e.target.closest(".dataset-option"); if (!b) return;
    active = +b.dataset.i;
    picker.querySelectorAll(".dataset-option").forEach(x => { const on = x === b; x.classList.toggle("is-selected", on); x.setAttribute("aria-pressed", on); });
    render();
  });
  document.querySelectorAll(".view-option").forEach(b => b.addEventListener("click", () => {
    view = b.dataset.view;
    document.querySelectorAll(".view-option").forEach(x => { const on = x === b; x.classList.toggle("is-selected", on); x.setAttribute("aria-pressed", on); });
    render();
  }));
  render();
})();
