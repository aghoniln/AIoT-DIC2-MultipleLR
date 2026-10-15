/**
 * MLR Studio 2.0 - Core Application Logic
 * Dual-Dataset Engine (50 Startups & Boston Housing)
 * Powered by Precomputed scikit-learn results in data.js
 */
(() => {
  const DATA = window.STARTUP_DATA;
  if (!DATA) {
    console.error("data.js not loaded");
    return;
  }

  // --- State Architecture ---
  const state = {
    problem: "startup", // "startup" | "boston"
    algo: "lasso",      // "lasso" | "forward" | "rfe" | "elasticnet" | "exhaustive"
    k: 2,
    compareAll: false,
    theme: localStorage.getItem("mlr_studio_theme") || "dark"
  };

  // Algorithm Metadata
  const ALGORITHMS = [
    { id: "lasso", name: "Lasso (L1)", icon: "fa-filter", tag: "Embedded L1", color: "#06b6d4" },
    { id: "forward", name: "Forward SFS", icon: "fa-arrow-trend-up", tag: "Greedy Step", color: "#10b981" },
    { id: "rfe", name: "RFE", icon: "fa-scissors", tag: "Backward Elim", color: "#8b5cf6" },
    { id: "elasticnet", name: "ElasticNet", icon: "fa-diagram-project", tag: "L1 + L2 Mix", color: "#f59e0b" },
    { id: "exhaustive", name: "Exhaustive", icon: "fa-magnifying-glass-chart", tag: "Best Subset", color: "#ec4899" }
  ];

  // Presets for What-If Simulator
  const SIM_PRESETS = {
    startup: [
      { name: "🚀 Unicorn Venture", values: { "R&D Spend": 165000, "Marketing Spend": 450000, "Administration": 130000, "State": "New York" } },
      { name: "🌱 Bootstrapped Lean", values: { "R&D Spend": 50000, "Marketing Spend": 20000, "Administration": 60000, "State": "California" } },
      { name: "📣 Marketing Heavy", values: { "R&D Spend": 90000, "Marketing Spend": 380000, "Administration": 110000, "State": "Florida" } }
    ],
    boston: [
      { name: "🏡 Suburban Luxury", values: { "rm": 7.8, "lstat": 4.5, "ptratio": 14.0, "dis": 5.2, "crim": 0.05, "tax": 220, "nox": 0.42 } },
      { name: "🏙 Urban Core", values: { "rm": 5.8, "lstat": 18.2, "ptratio": 20.2, "dis": 2.1, "crim": 4.5, "tax": 666, "nox": 0.68 } },
      { name: "🚆 Commuter Town", values: { "rm": 6.3, "lstat": 10.5, "ptratio": 17.5, "dis": 4.0, "crim": 0.8, "tax": 380, "nox": 0.51 } }
    ]
  };

  // Live simulation input values store
  const simValues = {
    startup: { "R&D Spend": 100000, "Marketing Spend": 150000, "Administration": 120000, "State": "New York" },
    boston: { "rm": 6.2, "lstat": 12.0, "ptratio": 18.0, "dis": 3.8, "crim": 0.5, "tax": 330, "nox": 0.52 }
  };

  // Cache DOM references
  const $ = id => document.getElementById(id);

  // Helper formatting
  const fmtMoney = (val, dec = 2) => {
    if (isNaN(val)) return "$0.00";
    const sign = val < 0 ? "-$" : "$";
    return sign + Math.abs(val).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
  };

  const fmtUnit = (val, unit = "$") => {
    if (unit === "$") return fmtMoney(val);
    return `${val.toFixed(2)} ${unit}`;
  };

  const getProblemData = () => DATA.problems[state.problem];

  const getActiveModel = () => {
    const pData = getProblemData();
    const algoKey = state.compareAll ? "lasso" : state.algo;
    return pData.results[algoKey][state.k - 1];
  };

  const bestKForAlgo = (algoKey) => {
    const list = getProblemData().results[algoKey];
    return list.reduce((bestIdx, item, idx) => (item.rmse < list[bestIdx].rmse ? idx : bestIdx), 0) + 1;
  };

  // Chart instances registry
  const chartInstances = {};

  // --- Theme Controller ---
  function applyTheme(t, updateCharts = false) {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("mlr_studio_theme", t);
    state.theme = t;
    const icon = $("themeIcon");
    if (icon) icon.className = t === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
    if (updateCharts) renderCharts();
  }
  $("themeToggle").onclick = () => applyTheme(state.theme === "dark" ? "light" : "dark", true);
  applyTheme(state.theme, false);

  // --- Problem / Dataset Switching ---
  $("btnProblemStartup").onclick = () => switchProblem("startup");
  $("btnProblemBoston").onclick = () => switchProblem("boston");

  function switchProblem(prob) {
    if (state.problem === prob) return;
    state.problem = prob;
    $("btnProblemStartup").classList.toggle("active", prob === "startup");
    $("btnProblemBoston").classList.toggle("active", prob === "boston");

    // Default k to best-k of default algo
    state.k = bestKForAlgo(state.algo);

    // Update download link
    const dlBtn = $("btnDownloadCsv");
    if (dlBtn) {
      dlBtn.href = prob === "startup" ? "assets/50_Startups.csv" : "assets/BostonHousing.csv";
      dlBtn.querySelector(".action-label").textContent = prob === "startup" ? "50_Startups.csv" : "BostonHousing.csv";
    }

    renderAll();
  }

  // --- Render Header & Briefing ---
  function renderBriefing() {
    const pData = getProblemData();
    const isStartup = state.problem === "startup";

    $("briefingDatasetTag").textContent = isStartup ? "Problem 1: Primary Upload" : "Problem 2: Workflow Transfer";
    $("briefingDatasetTag").className = `badge-pill ${isStartup ? "pill-cyan" : "pill-violet"}`;

    const bk = bestKForAlgo("lasso");
    const bestModel = pData.results.lasso[bk - 1];
    const fullModel = pData.results.lasso[pData.results.lasso.length - 1];
    const unit = pData.eda.target_unit;

    $("briefingSweetTag").textContent = `Sweet Spot: k=${bk} (${fmtUnit(bestModel.rmse, unit)} RMSE)`;
    $("briefingTitle").textContent = pData.title;

    $("briefingDesc").innerHTML = isStartup
      ? `Comparative multiple linear regression benchmarks across 5 feature selection strategies (<strong>Lasso</strong>, <strong>Forward SFS</strong>, <strong>RFE</strong>, <strong>ElasticNet</strong>, <strong>Exhaustive</strong>). Pinpoints the <strong>2-feature sweet spot</strong> (R&amp;D + Marketing) vs. the <strong>5-feature overfitting cliff</strong> (Admin + State dummies).`
      : `Demonstrating cross-domain transfer of the identical <strong>CRISP-DM ML workflow</strong> to the Boston Housing dataset. Evaluating top structural and socio-economic predictors (<em>RM</em>, <em>LSTAT</em>, <em>PTRATIO</em>, <em>DIS</em>, <em>CRIM</em>) to predict Median Home Value.`;

    $("statRowsCols").textContent = `${pData.eda.total_rows} × ${pData.features.length + 1}`;
    $("statSplit").textContent = `${pData.eda.n_train} Train / ${pData.eda.n_test} Test`;
    $("statSweetModel").textContent = `k = ${bk}`;
    $("statSweetDetail").textContent = `Min RMSE: ${fmtUnit(bestModel.rmse, unit)}`;

    const overfitDiff = fullModel.rmse - bestModel.rmse;
    $("statOverfit").textContent = `k = 5`;
    $("statOverfitDetail").textContent = `Error Spike: +${fmtUnit(overfitDiff, unit)}`;
  }

  // --- Render Algorithm Selector ---
  function renderAlgorithmSelector() {
    const container = $("algoDeck");
    container.innerHTML = ALGORITHMS.map(algo => `
      <button class="algo-card-btn ${!state.compareAll && state.algo === algo.id ? "active" : ""}" data-algo="${algo.id}">
        <div class="algo-top-row">
          <span class="algo-name"><i class="fa-solid ${algo.icon}" style="color:${algo.color}"></i> ${algo.name}</span>
          <span class="algo-tag">${algo.tag}</span>
        </div>
      </button>
    `).join("") + `
      <button class="btn-compare-all ${state.compareAll ? "active" : ""}" id="btnCompareAll">
        <i class="fa-solid fa-layer-group"></i> 🌟 Compare All 5 Methods Simultaneously
      </button>
    `;

    container.querySelectorAll(".algo-card-btn").forEach(btn => {
      btn.onclick = () => {
        state.algo = btn.dataset.algo;
        state.compareAll = false;
        state.k = bestKForAlgo(state.algo);
        renderAll();
      };
    });

    $("btnCompareAll").onclick = () => {
      state.compareAll = !state.compareAll;
      renderAll();
    };
  }

  // --- Render Feature Stepper ---
  function renderFeatureStepper() {
    const container = $("kStepper");
    const pData = getProblemData();
    const bk = bestKForAlgo(state.algo);
    const algoResults = pData.results[state.algo];

    container.innerHTML = [1, 2, 3, 4, 5].map(k => {
      const isSweet = k === bk;
      const isCliff = k === 5;
      const feats = algoResults[k - 1].features;
      const shortName = k === 1 ? feats[0] : (isSweet ? "★ Sweet Spot" : (isCliff ? "⚠ Overfit Cliff" : `+ ${feats[k - 1]}`));

      let cls = "k-step-btn";
      if (state.k === k) cls += " active";
      if (isSweet) cls += " sweet";
      if (isCliff) cls += " cliff";

      return `
        <button class="${cls}" data-k="${k}">
          <span class="k-title">k = ${k}</span>
          <span class="k-sub" title="${feats.join(', ')}">${shortName}</span>
        </button>
      `;
    }).join("");

    container.querySelectorAll(".k-step-btn").forEach(btn => {
      btn.onclick = () => {
        state.k = parseInt(btn.dataset.k);
        renderMetrics();
        renderEquation();
        renderSimulator();
        renderTables();
        renderCharts();
      };
    });

    // Update active features rack
    const currentModel = getActiveModel();
    $("activeCount").textContent = currentModel.features.length;
    $("activeChipsList").innerHTML = currentModel.features.map(f => `
      <span class="feature-token">${f}</span>
    `).join("");

    const parTag = $("parsimonyTag");
    if (state.k === bk) {
      parTag.className = "status-chip chip-green";
      parTag.innerHTML = `<i class="fa-solid fa-star"></i> Optimal Sweet Spot (k=${bk})`;
    } else if (state.k === 5) {
      parTag.className = "status-chip chip-amber";
      parTag.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Overfitting Warning`;
    } else {
      parTag.className = "status-chip chip-cyan";
      parTag.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${state.k} Features Active`;
    }
  }

  // --- Render Equation ---
  function renderEquation() {
    const model = getActiveModel();
    const algoMeta = ALGORITHMS.find(a => a.id === state.algo);
    $("equationModelTag").textContent = `${algoMeta.name} · k=${state.k}`;

    const cs = model.coef_stats;
    const interceptVal = cs[0].coef;
    const targetLabel = state.problem === "startup" ? "Profit" : "MEDV";

    let formulaParts = [`<span class="eq-term-intercept">${interceptVal.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>`];

    cs.slice(1).forEach(c => {
      const sign = c.coef >= 0 ? `<span class="eq-term-pos">+</span>` : `<span class="eq-term-neg">−</span>`;
      const absVal = Math.abs(c.coef).toFixed(4);
      formulaParts.push(`${sign} ${absVal} · <span class="eq-term-var">[${c.feature}]</span>`);
    });

    $("equationRender").innerHTML = `<strong>${targetLabel}</strong> = ${formulaParts.join(" ")}`;
  }

  // --- Render KPI Ribbon ---
  function renderMetrics() {
    const model = getActiveModel();
    const bk = bestKForAlgo(state.algo);
    const isBest = state.k === bk;
    const isCliff = state.k === 5;
    const unit = getProblemData().eda.target_unit;

    const pText = model.f_p < 1e-4 ? "p < 0.0001" : `p = ${model.f_p.toFixed(4)}`;

    $("kpiRibbon").innerHTML = `
      <div class="kpi-card ${isBest ? 'sweet' : (isCliff ? 'cliff' : '')}">
        <div class="kpi-header">
          <span class="kpi-name">Holdout Test RMSE</span>
          <span class="status-chip ${isBest ? 'chip-green' : (isCliff ? 'chip-amber' : 'chip-cyan')}">
            ${isBest ? '★ Minimum' : (isCliff ? '⚠ Overfit' : 'k = ' + state.k)}
          </span>
        </div>
        <div class="kpi-number">${fmtUnit(model.rmse, unit)}</div>
        <div class="kpi-footer">Out-of-sample root mean squared error</div>
      </div>

      <div class="kpi-card ${isBest ? 'sweet' : ''}">
        <div class="kpi-header">
          <span class="kpi-name">R² Variance Fit</span>
          <span class="status-chip chip-cyan">${(model.r2 * 100).toFixed(1)}% Fit</span>
        </div>
        <div class="kpi-number">${model.r2.toFixed(4)}</div>
        <div class="kpi-footer">Explains ${(model.r2 * 100).toFixed(2)}% of variance</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-name">Adjusted R²</span>
          <span class="status-chip chip-violet">Penalized</span>
        </div>
        <div class="kpi-number">${model.adj_r2.toFixed(4)}</div>
        <div class="kpi-footer">Accounts for degrees of freedom</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-name">Mean Absolute Error</span>
          <span class="status-chip chip-cyan">Avg Error</span>
        </div>
        <div class="kpi-number">${fmtUnit(model.mae, unit)}</div>
        <div class="kpi-footer">Average absolute prediction deviation</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-header">
          <span class="kpi-name">F-Statistic</span>
          <span class="status-chip ${model.f_p < 0.05 ? 'chip-green' : 'chip-amber'}">${pText}</span>
        </div>
        <div class="kpi-number">${model.f_stat.toFixed(1)}</div>
        <div class="kpi-footer">${model.f_p < 0.05 ? 'Statistically significant overall' : 'Non-significant'}</div>
      </div>
    `;
  }

  // --- Render Simulator (What-If Predictor) ---
  function renderSimulator() {
    const isStartup = state.problem === "startup";
    const vals = simValues[state.problem];
    const model = getActiveModel();

    $("simulatorTitle").innerHTML = isStartup
      ? `<i class="fa-solid fa-wand-magic-sparkles"></i> Venture Profit What-If Simulator`
      : `<i class="fa-solid fa-house-laptop"></i> Boston Property Valuation Simulator`;

    $("displayTargetLabel").textContent = isStartup ? "Predicted Net Profit" : "Predicted Median Home Value (MEDV)";

    // Render presets
    const presets = SIM_PRESETS[state.problem];
    $("presetButtons").innerHTML = presets.map((p, idx) => `
      <button class="btn-preset" data-idx="${idx}">${p.name}</button>
    `).join("");

    $("presetButtons").querySelectorAll(".btn-preset").forEach(b => {
      b.onclick = () => {
        const p = presets[parseInt(b.dataset.idx)];
        Object.assign(vals, p.values);
        updateSimulatorInputs();
        calculateSimulation();
      };
    });

    // Render dynamic inputs based on dataset
    const inputContainer = $("simulatorInputs");
    if (isStartup) {
      const inputs = [
        { f: "R&D Spend", id: "simRD", min: 0, max: 200000, step: 2500, unit: "$" },
        { f: "Marketing Spend", id: "simMkt", min: 0, max: 500000, step: 5000, unit: "$" },
        { f: "Administration", id: "simAdmin", min: 0, max: 250000, step: 2500, unit: "$" }
      ];

      inputContainer.innerHTML = inputs.map(inp => {
        const inModel = model.features.includes(inp.f);
        return `
          <div class="sim-input-row ${inModel ? '' : 'inactive'}" id="row_${inp.id}">
            <div class="sim-label-row">
              <span>${inp.f}</span>
              <span class="sim-status-tag ${inModel ? 'active' : ''}">${inModel ? 'In Model' : 'Inactive'}</span>
              <span class="sim-value-pill" id="lbl_${inp.id}">${fmtMoney(vals[inp.f], 0)}</span>
            </div>
            <input type="range" class="sim-slider" id="rng_${inp.id}" min="${inp.min}" max="${inp.max}" step="${inp.step}" value="${vals[inp.f]}" />
          </div>
        `;
      }).join("") + `
        <div class="sim-input-row" id="row_simState">
          <div class="sim-label-row">
            <span>State Location</span>
            <span class="sim-status-tag active">Dummy Coded</span>
          </div>
          <select id="sel_simState" class="studio-input">
            <option value="California" ${vals.State === 'California' ? 'selected' : ''}>California (Reference Base)</option>
            <option value="New York" ${vals.State === 'New York' ? 'selected' : ''}>New York</option>
            <option value="Florida" ${vals.State === 'Florida' ? 'selected' : ''}>Florida</option>
          </select>
        </div>
      `;

      inputs.forEach(inp => {
        const slider = $(`rng_${inp.id}`);
        slider.oninput = () => {
          vals[inp.f] = parseFloat(slider.value);
          $(`lbl_${inp.id}`).textContent = fmtMoney(vals[inp.f], 0);
          calculateSimulation();
        };
      });

      $("sel_simState").onchange = (e) => {
        vals.State = e.target.value;
        calculateSimulation();
      };

    } else {
      // Boston Housing inputs
      const bostonInputs = [
        { f: "rm", label: "Average Rooms (RM)", min: 3.5, max: 9.0, step: 0.1, fmt: v => v.toFixed(1) + " rooms" },
        { f: "lstat", label: "Lower Status Population % (LSTAT)", min: 1.0, max: 40.0, step: 0.5, fmt: v => v.toFixed(1) + "%" },
        { f: "ptratio", label: "Pupil-Teacher Ratio (PTRATIO)", min: 12.0, max: 23.0, step: 0.2, fmt: v => v.toFixed(1) + " : 1" },
        { f: "dis", label: "Distance to Employment (DIS)", min: 1.0, max: 12.0, step: 0.2, fmt: v => v.toFixed(1) + " mi" },
        { f: "crim", label: "Per Capita Crime Rate (CRIM)", min: 0.01, max: 20.0, step: 0.1, fmt: v => v.toFixed(2) }
      ];

      inputContainer.innerHTML = bostonInputs.map(inp => {
        const inModel = model.features.includes(inp.f);
        return `
          <div class="sim-input-row ${inModel ? '' : 'inactive'}" id="row_${inp.f}">
            <div class="sim-label-row">
              <span>${inp.label}</span>
              <span class="sim-status-tag ${inModel ? 'active' : ''}">${inModel ? 'In Model' : 'Inactive'}</span>
              <span class="sim-value-pill" id="lbl_${inp.f}">${inp.fmt(vals[inp.f])}</span>
            </div>
            <input type="range" class="sim-slider" id="rng_${inp.f}" min="${inp.min}" max="${inp.max}" step="${inp.step}" value="${vals[inp.f]}" />
          </div>
        `;
      }).join("");

      bostonInputs.forEach(inp => {
        const slider = $(`rng_${inp.f}`);
        slider.oninput = () => {
          vals[inp.f] = parseFloat(slider.value);
          $(`lbl_${inp.f}`).textContent = inp.fmt(vals[inp.f]);
          calculateSimulation();
        };
      });
    }

    calculateSimulation();
  }

  function updateSimulatorInputs() {
    const isStartup = state.problem === "startup";
    const vals = simValues[state.problem];

    if (isStartup) {
      $("rng_simRD").value = vals["R&D Spend"];
      $("lbl_simRD").textContent = fmtMoney(vals["R&D Spend"], 0);
      $("rng_simMkt").value = vals["Marketing Spend"];
      $("lbl_simMkt").textContent = fmtMoney(vals["Marketing Spend"], 0);
      $("rng_simAdmin").value = vals["Administration"];
      $("lbl_simAdmin").textContent = fmtMoney(vals["Administration"], 0);
      $("sel_simState").value = vals["State"];
    } else {
      ["rm", "lstat", "ptratio", "dis", "crim"].forEach(f => {
        const el = $(`rng_${f}`);
        if (el) {
          el.value = vals[f];
          $(`lbl_${f}`).textContent = vals[f];
        }
      });
    }
  }

  function calculateSimulation() {
    const model = getActiveModel();
    const isStartup = state.problem === "startup";
    const vals = simValues[state.problem];

    let prediction = model.intercept;
    let parts = [];

    if (isStartup) {
      const featureMap = {
        "R&D Spend": vals["R&D Spend"],
        "Marketing Spend": vals["Marketing Spend"],
        "Administration": vals["Administration"],
        "State_Florida": vals.State === "Florida" ? 1 : 0,
        "State_New York": vals.State === "New York" ? 1 : 0
      };

      Object.entries(model.coef).forEach(([feat, weight]) => {
        const v = featureMap[feat] || 0;
        const contrib = weight * v;
        prediction += contrib;
        if (v !== 0 || !feat.startsWith("State_")) {
          parts.push(`${feat.replace("State_", "")}: ${contrib >= 0 ? "+" : ""}${fmtMoney(contrib, 0)}`);
        }
      });

      $("displayTargetValue").textContent = fmtMoney(prediction);
      $("displayFormulaBreakdown").innerHTML = `
        <span>Intercept: ${fmtMoney(model.intercept, 0)}</span> · 
        <span class="breakdown">${parts.join(" · ")}</span>
      `;
    } else {
      Object.entries(model.coef).forEach(([feat, weight]) => {
        const v = vals[feat] || 0;
        const contrib = weight * v;
        prediction += contrib;
        parts.push(`${feat.toUpperCase()}: ${contrib >= 0 ? "+" : ""}${contrib.toFixed(2)}`);
      });

      $("displayTargetValue").textContent = `$${(prediction * 1000).toLocaleString("en-US", { maximumFractionDigits: 0 })} (${prediction.toFixed(2)}k)`;
      $("displayFormulaBreakdown").innerHTML = `
        <span>β₀: ${model.intercept.toFixed(2)}k</span> · 
        <span class="breakdown">${parts.join(" · ")}</span>
      `;
    }
  }

  // --- Render Tables ---
  function renderTables() {
    const model = getActiveModel();
    const pData = getProblemData();
    const unit = pData.eda.target_unit;

    // 1. Coefficients & VIF Table
    $("tableCoefBody").innerHTML = model.coef_stats.map(c => {
      const isIntercept = c.feature === "(Intercept)";
      const isSig = c.pvalue < 0.05;
      const pBadge = isSig
        ? `<span class="status-chip chip-green">${c.pvalue < 1e-4 ? '< 0.0001 ***' : c.pvalue.toFixed(4) + ' *'}</span>`
        : `<span class="status-chip chip-amber">${c.pvalue.toFixed(4)} (n.s.)</span>`;

      let vifHtml = `<span style="color:var(--text-muted)">–</span>`;
      if (!isIntercept && c.vif !== null) {
        vifHtml = c.vif < 5
          ? `<span class="status-chip chip-green">${c.vif.toFixed(2)} (Safe)</span>`
          : `<span class="status-chip chip-amber">${c.vif.toFixed(2)} (High)</span>`;
      }

      return `
        <tr>
          <td><strong>${c.feature}</strong></td>
          <td class="font-mono" style="font-weight:700;color:var(--text-primary)">${c.coef.toFixed(4)}</td>
          <td class="font-mono">${c.stderr.toFixed(4)}</td>
          <td class="font-mono">${c.tstat.toFixed(3)}</td>
          <td>${pBadge}</td>
          <td class="font-mono" style="font-size:12px">[${c.ci_lower.toFixed(2)}, ${c.ci_upper.toFixed(2)}]</td>
          <td>${vifHtml}</td>
        </tr>
      `;
    }).join("");

    // 2. Outliers Table
    $("tableOutliersBody").innerHTML = model.outliers.map(o => {
      const isHigh = o.abs_error > (state.problem === "startup" ? 10000 : 5.0);
      return `
        <tr>
          <td><strong>#${o.rank}</strong></td>
          <td>Sample #${o.index}</td>
          <td class="font-mono">${fmtUnit(o.actual, unit)}</td>
          <td class="font-mono">${fmtUnit(o.predicted, unit)}</td>
          <td class="font-mono" style="color:${o.residual >= 0 ? 'var(--emerald-primary)' : 'var(--rose-primary)'}">
            ${o.residual >= 0 ? '+' : ''}${fmtUnit(o.residual, unit)}
          </td>
          <td class="font-mono">${fmtUnit(o.abs_error, unit)}</td>
          <td>${o.pct_error.toFixed(2)}%</td>
          <td>
            <span class="status-chip ${isHigh ? 'chip-amber' : 'chip-green'}">
              ${isHigh ? 'High Residual' : 'In Envelope'}
            </span>
          </td>
        </tr>
      `;
    }).join("");

    // 3. Comparison Matrix
    const matrixEl = $("tableMatrix");
    matrixEl.innerHTML = `
      <thead>
        <tr>
          <th>Algorithm Strategy</th>
          <th>k = 1</th>
          <th>k = 2</th>
          <th>k = 3</th>
          <th>k = 4</th>
          <th>k = 5</th>
          <th>Optimal k</th>
        </tr>
      </thead>
      <tbody>
        ${ALGORITHMS.map(a => {
          const res = pData.results[a.id];
          const bk = bestKForAlgo(a.id);
          return `
            <tr>
              <td><i class="fa-solid ${a.icon}" style="color:${a.color}"></i> <strong>${a.name}</strong></td>
              ${[1, 2, 3, 4, 5].map(k => {
                const item = res[k - 1];
                const isSelected = !state.compareAll && state.algo === a.id && state.k === k;
                const isRowBest = k === bk;
                return `
                  <td class="matrix-cell ${isRowBest ? 'best' : ''} ${isSelected ? 'selected' : ''}" 
                      data-algo="${a.id}" data-k="${k}" title="Features: ${item.features.join(', ')}">
                    ${fmtUnit(item.rmse, unit)} ${isRowBest ? '★' : ''}
                  </td>
                `;
              }).join("")}
              <td><span class="badge-pill pill-emerald">k = ${bk}</span></td>
            </tr>
          `;
        }).join("")}
      </tbody>
    `;

    matrixEl.querySelectorAll(".matrix-cell").forEach(cell => {
      cell.onclick = () => {
        state.algo = cell.dataset.algo;
        state.k = parseInt(cell.dataset.k);
        state.compareAll = false;
        renderAll();
        window.scrollTo({ top: 320, behavior: "smooth" });
      };
    });

    // 4. Dataset Record Browser
    renderDatasetRecords();
  }

  function renderDatasetRecords(filterText = "") {
    const pData = getProblemData();
    const rows = pData.eda.rows;
    if (!rows || rows.length === 0) return;

    const cols = Object.keys(rows[0]);
    const q = filterText.toLowerCase();
    const filtered = rows.filter(r => !q || Object.values(r).some(v => String(v).toLowerCase().includes(q)));

    $("dataRecordSub").textContent = `Showing ${filtered.length} of ${pData.eda.total_rows} total observations. Train/Test split: 80% / 20%.`;

    $("tableDataRecords").innerHTML = `
      <thead>
        <tr>
          <th>#</th>
          ${cols.map(c => `<th>${c}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${filtered.map((r, i) => `
          <tr>
            <td>${i + 1}</td>
            ${cols.map(c => `<td class="font-mono">${typeof r[c] === 'number' ? r[c].toLocaleString() : r[c]}</td>`).join("")}
          </tr>
        `).join("")}
      </tbody>
    `;
  }
  $("searchInput").oninput = (e) => renderDatasetRecords(e.target.value);

  // --- Render CRISP-DM Framework ---
  function renderCrispFlow() {
    const pData = getProblemData();
    const isStartup = state.problem === "startup";

    const crispData = [
      {
        num: 1, title: "Business Understanding",
        items: isStartup
          ? ["Objective: Predict venture profit to guide VC funding & budget allocation", "Stakeholders: Investors, Founders, Executives", "Success metric: R² > 0.90, Test RMSE < $10,000"]
          : ["Objective: Assess property valuations across Boston metropolitan areas", "Stakeholders: Urban planners, Buyers, Tax assessors", "Success metric: Low prediction bias & stable generalization"]
      },
      {
        num: 2, title: "Data Understanding",
        items: isStartup
          ? ["50 Startups dataset (5 features, 0 missing values)", "R&D Spend exhibits highest correlation with Profit (r = 0.97)", "Administration spend has negligible linear relationship (r = 0.20)"]
          : ["506 Census tracts (13 predictors, 0 missing values)", "LSTAT (-0.74) and RM (+0.70) show strongest correlation with MEDV", "Socio-economic factors dominate pricing"]
      },
      {
        num: 3, title: "Data Preparation",
        items: isStartup
          ? ["One-hot dummy encoding for State (drop='first' to prevent dummy variable trap)", "Standard scaling applied to predictors for regularized L1/L2 ranking", "80/20 train-test split (random_state = 0)"]
          : ["Standardize continuous variables for penalized regressions", "Check multicollinearity across RAD & TAX predictors", "80/20 holdout train-test split"]
      },
      {
        num: 4, title: "Modeling & Selection",
        items: isStartup
          ? ["Execute 5 algorithms: Lasso, Forward SFS, RFE, ElasticNet, Exhaustive", "Evaluate models across feature budgets k = 1 to 5", "Ordinary Least Squares parameter estimation"]
          : ["Directly applied identical feature-selection skill & workflow pipeline", "Progressively budget features k = 1 to 5", "Automated OLS regression fitting"]
      },
      {
        num: 5, title: "Evaluation & Parsimony",
        items: isStartup
          ? ["k = 2 model minimizes test error (RMSE: $8,198.80)", "k = 5 model overfits test data (RMSE: $9,137.99)", "Occam's Razor confirms parsimonious model superiority"]
          : ["Top features (LSTAT + RM) capture the majority of price variance", "Dimensionality reduction protects out-of-sample stability", "Residual diagnostics confirm error envelope"]
      },
      {
        num: 6, title: "Deployment",
        items: [
          "Interactive Web Application deployed to GitHub Pages",
          "Model pipeline exported as serial joblib artifact",
          "Real-time what-if simulator for instant scenario forecasting"
        ]
      }
    ];

    $("crispFlowGrid").innerHTML = crispData.map(c => `
      <div class="crisp-step-card">
        <div class="crisp-step-num">${c.num}</div>
        <div class="crisp-step-title">${c.title}</div>
        <ul class="crisp-step-list">
          ${c.items.map(it => `<li>${it}</li>`).join("")}
        </ul>
      </div>
    `).join("");
  }

  // --- Render Chart.js Visualizations ---
  function createChart(canvasId, config) {
    if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
    chartInstances[canvasId] = new Chart($(canvasId), config);
  }

  function renderCharts() {
    if (!window.Chart) return;

    const pData = getProblemData();
    const isDark = state.theme === "dark";
    const gridColor = isDark ? "rgba(255, 255, 255, 0.06)" : "#f1f5f9";
    const textColor = isDark ? "#94a3b8" : "#475569";
    const unit = pData.eda.target_unit;

    Chart.defaults.color = textColor;
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // 1. Dual Complexity Charts (RMSE & R2)
    const algosToDraw = state.compareAll ? ALGORITHMS : [ALGORITHMS.find(a => a.id === state.algo)];
    const kLabels = ["k=1", "k=2", "k=3", "k=4", "k=5"];

    const rmseDatasets = algosToDraw.map(algo => ({
      label: algo.name,
      data: pData.results[algo.id].map(r => r.rmse),
      borderColor: algo.color,
      backgroundColor: algo.color,
      borderWidth: 2.5,
      tension: 0.15,
      pointRadius: [1, 2, 3, 4, 5].map(k => (!state.compareAll && k === state.k ? 8 : (k === bestKForAlgo(algo.id) ? 6 : 4))),
      pointBackgroundColor: [1, 2, 3, 4, 5].map(k => (k === bestKForAlgo(algo.id) ? "#10b981" : (k === 5 ? "#f59e0b" : algo.color))),
      pointBorderColor: "#fff",
      pointBorderWidth: 2
    }));

    const r2Datasets = algosToDraw.map(algo => ({
      label: algo.name,
      data: pData.results[algo.id].map(r => r.r2),
      borderColor: algo.color,
      backgroundColor: algo.color,
      borderWidth: 2.5,
      tension: 0.15,
      pointRadius: [1, 2, 3, 4, 5].map(k => (!state.compareAll && k === state.k ? 8 : (k === bestKForAlgo(algo.id) ? 6 : 4))),
      pointBackgroundColor: [1, 2, 3, 4, 5].map(k => (k === bestKForAlgo(algo.id) ? "#10b981" : algo.color)),
      pointBorderColor: "#fff",
      pointBorderWidth: 2
    }));

    createChart("chartRmse", {
      type: "line",
      data: { labels: kLabels, datasets: rmseDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: state.compareAll, position: "top" },
          tooltip: {
            callbacks: {
              label: c => `${c.dataset.label}: ${fmtUnit(c.parsed.y, unit)}`
            }
          }
        },
        scales: {
          x: { grid: { color: gridColor } },
          y: { grid: { color: gridColor }, title: { display: true, text: `RMSE (${unit}) (Lower is Better)` } }
        },
        onClick: (e, els) => {
          if (els.length > 0) {
            state.k = els[0].index + 1;
            renderAll();
          }
        }
      }
    });

    createChart("chartR2", {
      type: "line",
      data: { labels: kLabels, datasets: r2Datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: state.compareAll, position: "top" },
          tooltip: {
            callbacks: {
              label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(4)}`
            }
          }
        },
        scales: {
          x: { grid: { color: gridColor } },
          y: { grid: { color: gridColor }, title: { display: true, text: `R² Score (Higher is Better)` } }
        }
      }
    });

    // 2. Scatter Actual vs Predicted
    const currentModel = getActiveModel();
    const lo = Math.floor(Math.min(...currentModel.actual, ...currentModel.pred) * 0.9);
    const hi = Math.ceil(Math.max(...currentModel.actual, ...currentModel.pred) * 1.05);

    createChart("chartScatter", {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Test Holdout Samples",
            data: currentModel.actual.map((a, idx) => ({ x: a, y: currentModel.pred[idx] })),
            backgroundColor: "#06b6d4",
            borderColor: "#fff",
            borderWidth: 1.5,
            pointRadius: 6,
            pointHoverRadius: 9
          },
          {
            label: "Ideal Calibration (y = x)",
            type: "line",
            data: [{ x: lo, y: lo }, { x: hi, y: hi }],
            borderColor: "#f43f5e",
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: c => `Actual: ${fmtUnit(c.parsed.x, unit)} → Pred: ${fmtUnit(c.parsed.y, unit)}`
            }
          }
        },
        scales: {
          x: { min: lo, max: hi, grid: { color: gridColor }, title: { display: true, text: `Actual (${unit})` } },
          y: { min: lo, max: hi, grid: { color: gridColor }, title: { display: true, text: `Predicted (${unit})` } }
        }
      }
    });

    // 3. Residuals by Sample
    const outliersSorted = [...currentModel.outliers].sort((a, b) => a.index - b.index);
    createChart("chartResiduals", {
      type: "bar",
      data: {
        labels: outliersSorted.map(o => `#${o.index}`),
        datasets: [{
          label: "Residual Error",
          data: outliersSorted.map(o => o.residual),
          backgroundColor: outliersSorted.map(o => o.residual >= 0 ? "#10b981" : "#f43f5e"),
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, title: { display: true, text: "Sample ID" } },
          y: { grid: { color: gridColor }, title: { display: true, text: `Residual (${unit})` } }
        }
      }
    });

    // 4. Feature Correlation
    const corr = pData.eda.corr;
    const corrKeys = Object.keys(corr).filter(k => k !== pData.eda.target);
    createChart("chartCorr", {
      type: "bar",
      data: {
        labels: corrKeys,
        datasets: [{
          data: corrKeys.map(k => corr[k]),
          backgroundColor: corrKeys.map(k => corr[k] >= 0 ? "#06b6d4" : "#8b5cf6"),
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { min: -1, max: 1, grid: { color: gridColor }, title: { display: true, text: "Pearson r" } }
        }
      }
    });

    // 5. Extra EDA
    if (state.problem === "startup") {
      $("extraEdaTitle").textContent = "Average Profit by State";
      const sp = pData.eda.state_profit;
      createChart("chartExtraEda", {
        type: "bar",
        data: {
          labels: Object.keys(sp),
          datasets: [{
            data: Object.values(sp),
            backgroundColor: ["#10b981", "#06b6d4", "#ec4899"],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { grid: { color: gridColor }, title: { display: true, text: "Mean Profit ($)" } }
          }
        }
      });
    } else {
      $("extraEdaTitle").textContent = "Target Price Distribution (MEDV)";
      const medvValues = pData.eda.rows.map(r => r.medv);
      const bins = [10, 20, 30, 40, 50];
      const counts = [0, 0, 0, 0, 0];
      medvValues.forEach(v => {
        if (v < 15) counts[0]++;
        else if (v < 25) counts[1]++;
        else if (v < 35) counts[2]++;
        else if (v < 45) counts[3]++;
        else counts[4]++;
      });
      createChart("chartExtraEda", {
        type: "bar",
        data: {
          labels: ["< $15k", "$15k-$25k", "$25k-$35k", "$35k-$45k", "≥ $45k"],
          datasets: [{
            data: counts,
            backgroundColor: "#10b981",
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { grid: { color: gridColor }, title: { display: true, text: "Tract Frequency" } }
          }
        }
      });
    }
  }

  // --- Master Render Function ---
  function renderAll() {
    renderBriefing();
    renderAlgorithmSelector();
    renderFeatureStepper();
    renderMetrics();
    renderEquation();
    renderSimulator();
    renderTables();
    renderCrispFlow();
    renderCharts();
  }

  // Initialize
  renderAll();
  document.fonts?.ready.then(() => renderCharts());
})();
