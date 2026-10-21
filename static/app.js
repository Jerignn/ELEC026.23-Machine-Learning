// ShopSmart Analytics - dashboard front end
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const money = (v) => "$" + Math.round(v).toLocaleString();
const pct = (v) => (v * 100).toFixed(1) + "%";
const charts = {};
let DATA = null;

async function api(url, body) {
  const res = await fetch(url, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  return res.json();
}

// ---------- Chart.js defaults (recessive grid/axes, thin marks) ----------
function themeCharts() {
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--grid");
  Chart.defaults.plugins.legend.display = false;   // HTML legends above each chart
  Chart.defaults.plugins.tooltip.backgroundColor = css("--text");
  Chart.defaults.plugins.tooltip.titleColor = css("--surface");
  Chart.defaults.plugins.tooltip.bodyColor = css("--surface");
  Chart.defaults.maintainAspectRatio = false;
}

function draw(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), config);
}

// history (solid) + forecast (dashed) line chart
function lineForecast(id, labels, hist, fut, fmt) {
  const c = css("--series-1");
  const histData = [...hist, ...fut.map(() => null)];
  const futData = [...hist.map(() => null), ...fut];
  futData[hist.length - 1] = hist[hist.length - 1]; // connect the two segments
  draw(id, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Actual", data: histData, borderColor: c, borderWidth: 2, pointRadius: 0, tension: 0.25 },
        { label: "Forecast", data: futData, borderColor: c, borderWidth: 2, borderDash: [6, 4],
          pointRadius: 3, pointBackgroundColor: c, tension: 0.25 },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      plugins: { tooltip: { filter: (i) => i.raw !== null && !(i.datasetIndex === 1 && i.dataIndex === hist.length - 1),
                            callbacks: { label: (i) => `${i.dataset.label}: ${fmt(i.raw)}` } } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { ticks: { callback: fmt }, beginAtZero: false },
      },
    },
  });
}

// ---------- KPIs ----------
function renderKpis(k, revenue) {
  $("k-rev").textContent = money(k.revenue_last_month);
  $("k-next").textContent = money(k.revenue_next_month);
  const d = (k.revenue_next_month - k.revenue_last_month) / k.revenue_last_month;
  $("k-delta").textContent = `${d >= 0 ? "▲" : "▼"} ${Math.abs(d * 100).toFixed(1)}% vs last month (${revenue.forecast[0].month})`;
  $("k-delta").className = "delta " + (d >= 0 ? "up" : "down");
  $("k-pos").textContent = k.positive_review_pct + "%";
  $("k-buy").textContent = k.likely_buyers.toLocaleString();
  $("k-buy-sub").textContent = `of ${k.customers.toLocaleString()} customers`;
  $("k-promo").textContent = k.promo_responders.toLocaleString();
  $("k-promo-sub").textContent = `${(k.promo_responders / k.customers * 100).toFixed(0)}% of customers to target`;
  $("k-ann").textContent = k.avg_purchase_prob + "%";
}

// ---------- 1. Linear Regression ----------
function renderRevenue(r) {
  const labels = [...r.history.map((h) => h.month), ...r.forecast.map((f) => f.month)];
  lineForecast("revChart", labels, r.history.map((h) => h.revenue), r.forecast.map((f) => f.revenue), money);
}

async function loadForecast() {
  const pid = $("productSel").value;
  const spend = $("spendRange").value;
  const f = await api(`/api/forecast/${pid}?ad_spend=${spend}`);
  const labels = [...f.history.map((h) => h.month), ...f.forecast.map((x) => x.month)];
  lineForecast("unitChart", labels, f.history.map((h) => h.units), f.forecast.map((x) => x.units),
    (v) => Math.round(v).toLocaleString());
  const total = f.forecast.reduce((s, x) => s + x.units, 0);
  $("forecastTable").innerHTML =
    `<tr><th>Month</th>${f.forecast.map((x) => `<th>${x.month}</th>`).join("")}</tr>` +
    `<tr><td>Units</td>${f.forecast.map((x) => `<td>${x.units}</td>`).join("")}</tr>` +
    `<tr><td>Revenue</td>${f.forecast.map((x) => `<td>${money(x.revenue)}</td>`).join("")}</tr>`;
  const c = f.coefficients;
  $("coefText").textContent =
    `Stock tip: plan ~${total.toLocaleString()} units for the next 6 months. ` +
    `Model: +${c.t} units/month trend, +${c.holiday} in Nov–Dec, +${(c.ad_spend * 1000).toFixed(1)} units per $1,000 ads.`;
}

// ---------- 2. Naive Bayes ----------
function renderReviews(rv) {
  const cats = rv.by_category;
  const gap = css("--surface");
  draw("sentChart", {
    type: "bar",
    data: {
      labels: cats.map((c) => c.category),
      datasets: [
        { label: "Positive", data: cats.map((c) => c.positive), backgroundColor: css("--pos"),
          borderColor: gap, borderWidth: { right: 2 }, borderSkipped: false, borderRadius: 4 },
        { label: "Negative", data: cats.map((c) => c.negative), backgroundColor: css("--neg"),
          borderColor: gap, borderWidth: { left: 2 }, borderSkipped: false, borderRadius: 4 },
      ],
    },
    options: {
      indexAxis: "y",
      scales: { x: { stacked: true, title: { display: true, text: "Reviews" } },
                y: { stacked: true, grid: { display: false } } },
      plugins: { tooltip: { mode: "index", callbacks: {
        footer: (items) => { const t = items.reduce((s, i) => s + i.raw, 0);
                             return `${pct(items[0].raw / t)} positive`; } } } },
    },
  });
  $("posWords").innerHTML = rv.top_words.positive.map((w) => `<span class="chip">${w}</span>`).join("");
  $("negWords").innerHTML = rv.top_words.negative.map((w) => `<span class="chip">${w}</span>`).join("");
  $("recentReviews").innerHTML = rv.recent.map((r) =>
    `<li><span class="tag ${r.predicted}">${r.predicted}</span><span>${r.text}</span></li>`).join("");
}

async function classifyReview() {
  const r = await api("/api/review", { text: $("reviewText").value });
  if (r.error) { $("reviewResult").textContent = r.error; return; }
  const good = r.label === "positive";
  $("reviewResult").innerHTML =
    `<div class="verdict ${good ? "yes" : "no"}">${good ? "👍 Positive" : "👎 Negative"} review</div>
     <div class="small muted">Confidence ${pct(r.confidence)} · P(positive) ${pct(r.probabilities.positive)}</div>
     <div class="meter"><div style="width:${r.probabilities.positive * 100}%"></div></div>`;
}

// ---------- 3. KNN ----------
const productLine = (p, extra) =>
  `<li><b>${p.name}</b> <span class="meta">· ${p.category} · $${p.price.toFixed(2)}${extra ? " · " + extra : ""}</span></li>`;

async function loadRecs() {
  const r = await api(`/api/recommend/${$("customerSel").value}`);
  $("ratedList").innerHTML = r.rated.map((p) => `<span class="chip">${p.name} ${"★".repeat(p.rating)}</span>`).join("");
  $("recList").innerHTML = r.recommendations.map((p) => productLine(p, `predicted rating ${p.score}`)).join("");
  $("neighList").innerHTML = r.neighbors.map((n) =>
    `<span class="chip">#${n.customer_id} · ${(n.similarity * 100).toFixed(0)}% similar</span>`).join("");
}

async function loadSimilar() {
  const s = await api(`/api/similar/${$("simProductSel").value}`);
  $("simList").innerHTML = s.map((p) => productLine(p, `${(p.similarity * 100).toFixed(0)}% similar`)).join("");
}

// ---------- 4-6. SVM / Decision Tree / ANN forms ----------
const FORMS = {
  svmForm: {
    url: "/api/predict/svm", out: "svmResult",
    fields: [["age", "Age", 29], ["income", "Income ($)", 52000], ["tenure_months", "Months as customer", 14],
             ["past_purchases", "Past purchases", 6], ["sessions_per_month", "Visits / month", 9],
             ["cart_adds", "Cart adds", 3], ["days_since_last_visit", "Days since visit", 4]],
    show: (r) => `<div class="verdict ${r.will_purchase ? "yes" : "no"}">${r.will_purchase ? "✔" : "✖"} ${r.label}</div>
                  <div class="small muted">Distance from SVM boundary: ${r.margin}</div>`,
  },
  treeForm: {
    url: "/api/predict/tree", out: "treeResult",
    fields: [["age", "Age", 26], ["income", "Income ($)", 32000], ["email_open_rate", "Email open rate (0–1)", 0.6],
             ["used_coupon_before", "Used coupon (0/1)", 1], ["days_since_last_visit", "Days since visit", 6],
             ["past_purchases", "Past purchases", 3], ["tenure_months", "Months as customer", 10]],
    show: (r) => `<div class="verdict ${r.will_respond ? "yes" : "no"}">${r.will_respond ? "✔ Send the promotion" : "✖ Unlikely to respond"}</div>
                  <div class="small muted">Response probability ${pct(r.probability)}</div>
                  <ol class="steps">${r.path.map((s) => `<li>${s}</li>`).join("")}</ol>`,
  },
  annForm: {
    url: "/api/predict/ann", out: "annResult",
    fields: [["sessions_per_month", "Visits / month", 8], ["avg_session_min", "Avg minutes / visit", 9],
             ["pages_per_session", "Pages / visit", 7], ["cart_adds", "Cart adds", 3],
             ["wishlist_items", "Wishlist items", 2], ["days_since_last_visit", "Days since visit", 3],
             ["past_purchases", "Past purchases", 4], ["email_open_rate", "Email open rate (0–1)", 0.4]],
    show: (r) => `<div class="verdict">${pct(r.probability)} · ${r.label}</div>
                  <div class="meter"><div style="width:${r.probability * 100}%"></div></div>`,
  },
};

function buildForms() {
  for (const [id, f] of Object.entries(FORMS)) {
    $(id).innerHTML = f.fields.map(([k, label, v]) =>
      `<label>${label}<input type="number" step="any" name="${k}" value="${v}"></label>`).join("");
    $(id).addEventListener("submit", (e) => { e.preventDefault(); submitForm(id); });
  }
  document.querySelectorAll("button[data-form]").forEach((b) =>
    b.addEventListener("click", () => submitForm(b.dataset.form)));
}

async function submitForm(id) {
  const f = FORMS[id];
  const body = Object.fromEntries([...new FormData($(id))].map(([k, v]) => [k, Number(v)]));
  $(f.out).innerHTML = f.show(await api(f.url, body));
}

function renderInsights(d) {
  const imp = d.tree.importances;
  draw("impChart", {
    type: "bar",
    data: { labels: imp.map((i) => i.feature.replaceAll("_", " ")),
            datasets: [{ label: "Importance", data: imp.map((i) => i.importance),
                         backgroundColor: css("--series-1"), borderRadius: 4, barThickness: 16 }] },
    options: { indexAxis: "y", scales: { x: { max: 1, ticks: { callback: (v) => pct(v) } }, y: { grid: { display: false } } },
               plugins: { tooltip: { callbacks: { label: (i) => `Importance ${pct(i.raw)}` } } } },
  });
  $("treeRules").textContent = d.tree.rules;

  const h = d.ann.histogram;
  draw("annChart", {
    type: "bar",
    data: { labels: h.map((b) => b.bucket),
            datasets: [{ label: "Customers", data: h.map((b) => b.count), backgroundColor: css("--series-1"),
                         borderColor: css("--surface"), borderWidth: { left: 1, right: 1 }, borderSkipped: false,
                         borderRadius: { topLeft: 4, topRight: 4 } }] },
    options: { scales: { x: { grid: { display: false }, title: { display: true, text: "Purchase probability" } },
                         y: { title: { display: true, text: "Customers" } } },
               plugins: { tooltip: { callbacks: { label: (i) => `${i.raw} customers` } } } },
  });

  $("topTable").innerHTML =
    `<tr><th>Customer</th><th>Age</th><th>Favorite category</th><th>ANN purchase prob.</th>
         <th>SVM says</th><th>Promo response prob.</th><th>Action</th></tr>` +
    d.top_customers.map((c) => `<tr><td>#${c.customer_id}</td><td>${c.age}</td><td>${c.favorite_category}</td>
      <td>${pct(c.ann_prob)}</td><td>${c.svm_purchase ? "Will buy" : "Won't buy"}</td><td>${pct(c.promo_prob)}</td>
      <td>${c.promo_prob >= 0.5 ? "Send promo" : "Recommend products"}</td></tr>`).join("");
}

function renderMetrics(m) {
  const cols = ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const purpose = {
    "Linear Regression": "Predict future product sales",
    "Naive Bayes": "Classify reviews positive / negative",
    "SVM": "Classify customers likely to purchase",
    "Decision Tree": "Customers who respond to a promotion",
    "ANN": "Purchase probability from behavior",
  };
  const fmt = (v) => (v === undefined ? "—" : v.toFixed(3));
  let html = `<tr><th>Model</th><th>Used for</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>`;
  for (const [name, s] of Object.entries(m)) {
    if (name === "Linear Regression") {
      html += `<tr><td><b>${name}</b></td><td>${purpose[name]}</td><td colspan="5">R² = ${s.avg_r2} · mean abs. error = ${s.avg_mae_units} units / month</td></tr>`;
    } else {
      html += `<tr><td><b>${name}</b></td><td>${purpose[name]}</td>${cols.map((c) => `<td>${fmt(s[c])}</td>`).join("")}</tr>`;
    }
  }
  html += `<tr><td><b>KNN</b></td><td>Recommend similar products</td><td colspan="5">Unsupervised neighbor search (k = 10, cosine) — see recommendations above</td></tr>`;
  $("metricTable").innerHTML = html;
}

// ---------- boot ----------
async function init() {
  themeCharts();
  DATA = await api("/api/dashboard");
  renderKpis(DATA.kpis, DATA.revenue);
  renderRevenue(DATA.revenue);
  renderReviews(DATA.reviews);
  renderInsights(DATA);
  renderMetrics(DATA.metrics);

  const opts = DATA.products.map((p) => `<option value="${p.product_id}">${p.name} (${p.category})</option>`).join("");
  $("productSel").innerHTML = opts;
  $("simProductSel").innerHTML = opts;
  $("customerSel").innerHTML = DATA.knn_customers.map((c) => `<option value="${c}">Customer #${c}</option>`).join("");

  $("productSel").addEventListener("change", loadForecast);
  $("spendRange").addEventListener("input", () => { $("spendVal").textContent = money($("spendRange").value); });
  $("spendRange").addEventListener("change", loadForecast);
  $("customerSel").addEventListener("change", loadRecs);
  $("simProductSel").addEventListener("change", loadSimilar);
  $("reviewBtn").addEventListener("click", classifyReview);
  buildForms();

  loadForecast(); loadRecs(); loadSimilar(); classifyReview();
  Object.keys(FORMS).forEach(submitForm);
}

init();
