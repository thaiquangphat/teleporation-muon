// Muon Teleportation site: theme, hero animation, training demo, and data-driven results.

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
  


    // Hero: a Muon run that teleports along a loss contour, then makes larger steps.
    (() => {
      const svg = document.getElementById("heroSvg"), NS = "http://www.w3.org/2000/svg";
      const C = { x: 560, y: 200 }, A = -0.2, rx = 70, ry = 34;
      const P = (k, t) => { t *= Math.PI / 180; const x = k * rx * Math.cos(t), y = k * ry * Math.sin(t); return [C.x + x * Math.cos(A) - y * Math.sin(A), C.y + x * Math.sin(A) + y * Math.cos(A)]; };
      const mk = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      [1, 2, 3, 4, 4.1, 5].forEach((k, i) => { if (k !== 4.1) mk("ellipse", { cx: C.x, cy: C.y, rx: k * rx, ry: k * ry, transform: `rotate(${A * 180 / Math.PI} ${C.x} ${C.y})`, class: "contour" + (k === 1 ? " min" : "") }); });
      const seq = [], add = (el, gap) => { el.classList.add("h"); seq.push([el, gap]); return el; };
      const dot = (p, r = 5, c = "dot") => add(mk("circle", { cx: p[0], cy: p[1], r, class: c }), 0);
      const seg = (a, b) => add(mk("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "seg" }), 330);
      const one = [[5, 186], [4.62, 188], [4.3, 190], [4.05, 192]].map(n => P(...n));
      const two = [[4.05, 100], [3.0, 98], [1.9, 96], [0.9, 94], [0.08, 92]].map(n => P(...n));
      dot(one[0]); for (let i = 1; i < one.length; i++) { seg(one[i - 1], one[i]); dot(one[i]); }
      const arc = []; for (let t = 192; t >= 100; t -= 4) arc.push(P(4.05, t));
      const hop = add(mk("path", { d: "M" + arc.map(p => p.join(",")).join("L"), class: "hop", pathLength: 1 }), 500);
      dot(two[0], 7, "tdot"); seq[seq.length - 1][1] = 1000;
      for (let i = 1; i < two.length; i++) { seg(two[i - 1], two[i]); dot(two[i], i === two.length - 1 ? 6 : 5); }
      const text = (p, s, c) => { const t = mk("text", { x: p[0], y: p[1], class: "lbl " + (c || "") }); t.textContent = s; return t; };
      text([40, 252], "Small steps");
      text([70, 336], "Teleport along the contour", "t");
      text([590, 300], "Larger steps, same optimizer");
      text([C.x + 44, C.y - 44], "minimum");
      const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
      let timers = [];
      const play = () => {
        let t = 400;
        seq.forEach(([el, gap]) => { t += gap || 330; timers.push(setTimeout(() => el.classList.add("on"), t)); });
        timers.push(setTimeout(() => { seq.forEach(([el]) => el.classList.remove("on")); timers = []; play(); }, t + 3200));
      };
      if (reduce) seq.forEach(([el]) => el.classList.add("on")); else play();
    })();
  
(() => {
  let DATA;
  function initCharts() {
  
    const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    // Builds the stat strip, results table and dataset buttons from data/results.json.
    function buildPage() {
      const factsEl = document.getElementById("facts"); // optional: remove #facts from index.html to hide the strip
    if (factsEl) factsEl.innerHTML = (DATA.facts || []).map(f => `<div><b>${esc(f.value)}</b><span>${esc(f.label)}</span></div>`).join("");
      let h = '<table class="results-table"><thead><tr><th>Encoding</th><th>Teleport</th><th>Accuracy ↑</th><th>Epoch speedup ↑</th><th>Wall-clock speedup ↑</th><th>Time/epoch ↓</th></tr></thead><tbody>';
      DATA.datasets.forEach(d => {
        h += `<tr class="dataset-row"><th colspan="6">${esc(d.name)}</th></tr>`;
        const groups = [];
        d.rows.forEach(r => { const g = groups[groups.length - 1]; if (g && g.enc === r.encoding) g.rows.push(r); else groups.push({ enc: r.encoding, rows: [r] }); });
        groups.forEach(g => g.rows.forEach((r, i) => {
          h += `<tr${i ? "" : ' class="pe-start"'}>` + (i ? "" : `<th rowspan="${g.rows.length}">${esc(g.enc)}</th>`) +
            `<td style="text-align:left">${r.teleport ? "Yes" : "No"}</td><td${r.best ? ' class="best"' : ""}>${esc(r.accuracy)}</td><td>${esc(r.epochSpeedup || "—")}</td><td>${esc(r.wallSpeedup || "—")}</td><td>${esc(r.timePerEpoch || "—")}</td></tr>`;
        }));
      });
      document.getElementById("resultsTable").innerHTML = h + "</tbody></table>";
      document.getElementById("datasetPicker").innerHTML = DATA.datasets.map((d, i) =>
        `<button class="dataset-option${i ? "" : " is-selected"}" type="button" data-dataset="${i}" aria-pressed="${!i}"><span class="dataset-option-name">${esc(d.name)}</span><span class="dataset-option-meta"></span></button>`).join("");
    }
  
    buildPage();
    const curveData = { epoch: DATA.datasets.map(d => d.curves.epoch), wall: DATA.datasets.map(d => d.curves.wall || d.curves.epoch) };
    const DATASETS = DATA.datasets.map(d => d.name);
  
      // Per-configuration validation accuracy (%), mean over five seeds with min–max bands.
      // epoch view: x = epoch. wall view: x = mean logged process time (s) at which each epoch finished.
              const state = { view: "epoch", dataset: 0 };
  
      function seriesFor(view, dataset) { return curveData[view][dataset]; }
      function makeDatasets(seriesList) {
        const datasets = [];
        seriesList.forEach((series) => {
          const isApe = series.label.startsWith("APE");
          const color = getComputedStyle(document.documentElement).getPropertyValue(isApe ? "--ape" : "--rope").trim();
          const key = series.label.replaceAll(" ", "-").toLowerCase();
          const order = series.label.endsWith("no teleport") ? 1 : 0;
          datasets.push({ label: key + "-lower", data: series.lower, parsing: false, borderWidth: 0, pointRadius: 0, pointHitRadius: 0, backgroundColor: "transparent", fill: false, bandHelper: true, order });
          datasets.push({ label: key + "-range", data: series.upper, parsing: false, borderWidth: 0, pointRadius: 0, pointHitRadius: 0, backgroundColor: color + "26", fill: "-1", bandHelper: true, order });
          datasets.push({ label: series.label, data: series.mean, parsing: false, borderColor: color, backgroundColor: color, borderWidth: 2.2, borderDash: series.label.endsWith("no teleport") ? [6, 5] : [], pointRadius: 2, pointHoverRadius: 4, pointHitRadius: 12, tension: .18, order });
        });
        return datasets;
      }
      const chartCanvas = document.getElementById("trainingChart");
      const chart = new Chart(chartCanvas, {
        type: "line", data: { datasets: makeDatasets(seriesFor("epoch", 0)) },
        options: {
          responsive: true, maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false, axis: "x" },
          plugins: {
            legend: { position: "bottom", labels: { usePointStyle: true, pointStyle: "line", boxWidth: 22, padding: 14, color: "#526176", font: { family: "Bricolage Grotesque, system-ui, sans-serif", size: 10, weight: "600" }, filter: (item, data) => !data.datasets[item.datasetIndex].bandHelper, sort: (a, b) => a.datasetIndex - b.datasetIndex } },
            tooltip: { enabled: true, itemSort: (a, b) => a.datasetIndex - b.datasetIndex, filter: item => !item.dataset.bandHelper, backgroundColor: "#14243a", titleFont: { family: "monospace", size: 11 }, bodyFont: { family: "monospace", size: 11 }, padding: 11, cornerRadius: 6, displayColors: true,
              callbacks: {
                title: items => state.view === "epoch" ? "Epoch " + items[0].parsed.x : "Epoch " + items[0].raw.e + " · " + items[0].parsed.x.toFixed(1) + " s",
                label: ctx => " " + ctx.dataset.label + ": " + ctx.parsed.y.toFixed(3) + "%"
              } }
          },
          scales: {
            x: { type: "linear", min: 1, max: 10, title: { display: true, text: "Epoch", color: "#65758b", font: { size: 10 } }, ticks: { stepSize: 1, color: "#718096", font: { size: 9 } }, grid: { color: "rgba(89,112,142,.12)" } },
            y: { title: { display: true, text: "Validation accuracy (%)", color: "#65758b", font: { size: 10 } }, ticks: { color: "#718096", font: { size: 9 } }, grid: { color: "rgba(89,112,142,.12)" } }
          }
        }
      });
  
      const totalTime = d => Math.max(...seriesFor("wall", d).map(s => s.mean[s.mean.length - 1].x));
      function render() {
        const { view, dataset } = state;
        const list = seriesFor(view, dataset);
        const epochs = curveData.epoch[dataset][0].mean.length;
        const xs = chart.options.scales.x;
        chart.data.datasets = makeDatasets(list);
        if (view === "epoch") {
          xs.min = 1; xs.max = epochs; xs.ticks.stepSize = epochs <= 15 ? 1 : 5; xs.title.text = "Epoch";
          chart.options.interaction.mode = "index";
        } else {
          const tmax = totalTime(dataset);
          xs.min = 0; xs.max = Math.ceil(tmax / 25) * 25; xs.ticks.stepSize = undefined; xs.title.text = "Wall-clock time (s)";
          chart.options.interaction.mode = "nearest";
        }
        document.getElementById("activeDatasetName").textContent = DATASETS[dataset];
        document.getElementById("activeDatasetEpochs").textContent = view === "epoch" ? epochs + " epochs" : "up to ≈ " + Math.round(totalTime(dataset)) + " s";
        chartCanvas.setAttribute("aria-label", DATASETS[dataset] + " validation accuracy versus " + (view === "epoch" ? "epoch" : "wall-clock time"));
        document.querySelectorAll(".dataset-option").forEach(b => {
          const i = Number(b.dataset.dataset), sel = i === dataset;
          b.classList.toggle("is-selected", sel); b.setAttribute("aria-pressed", String(sel));
          b.querySelector(".dataset-option-meta").textContent = view === "epoch" ? curveData.epoch[i][0].mean.length + " epochs" : "≈ " + Math.round(totalTime(i)) + " s";
        });
        document.querySelectorAll(".view-option").forEach(b => {
          const sel = b.dataset.view === view;
          b.classList.toggle("is-selected", sel); b.setAttribute("aria-pressed", String(sel));
        });
        chart.update();
      }
      document.querySelectorAll(".dataset-option").forEach(b => b.addEventListener("click", () => { state.dataset = Number(b.dataset.dataset); render(); }));
      document.querySelectorAll(".view-option").forEach(b => b.addEventListener("click", () => { state.view = b.dataset.view; render(); }));
  
      function updateChartTheme(dark) {
        const tick = dark ? "#a8b7cc" : "#718096";
        const title = dark ? "#c1cee0" : "#65758b";
        const grid = dark ? "rgba(160,180,210,.14)" : "rgba(89,112,142,.12)";
        const legend = dark ? "#c1cee0" : "#526176";
        const tooltipBg = dark ? "#e8eef8" : "#14243a";
        const tooltipText = dark ? "#14243a" : "#ffffff";
        Object.values(Chart.instances).forEach(chart => {
          chart.options.plugins.legend.labels.color = legend;
          chart.options.plugins.tooltip.backgroundColor = tooltipBg;
          chart.options.plugins.tooltip.titleColor = tooltipText;
          chart.options.plugins.tooltip.bodyColor = tooltipText;
          for (const axis of Object.values(chart.options.scales)) {
            axis.ticks.color = tick;
            axis.grid.color = grid;
            if (axis.title) axis.title.color = title;
          }
          chart.update("none");
        });
        if (typeof render === "function") render();
      }
      render();
      updateChartTheme(document.documentElement.dataset.theme === "dark");
    
    window.updateChartTheme = updateChartTheme;
  }
  const show = msg => { document.getElementById("resultsTable").innerHTML = '<p class="small-note" style="padding:16px">' + msg + "</p>"; };
  // Server: read data/results.json. Opened as a local file, browsers block fetch, so use data/results.js.
  const loadData = () => fetch("data/results.json")
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .catch(() => { if (window.MUON_RESULTS) return window.MUON_RESULTS; throw new Error("no data"); });
  loadData().then(
    d => { DATA = d; try { initCharts(); } catch (e) { console.error(e); show("Could not draw the results. See the browser console for details."); } },
    () => show("Could not load the results data. Check that data/results.json (or data/results.js) exists.")
  );
})();