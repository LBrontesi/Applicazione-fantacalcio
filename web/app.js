"use strict";

import { getJSON, postJSON, postForm } from "./api-client.js";

/* ------------------------------------------------------------------ */
/* Utilities                                                          */
/* ------------------------------------------------------------------ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );

const num = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
};

const int = (value) => {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return String(Math.round(n));
};

const pct = (value, digits = 0) => {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
};

const debounce = (fn, ms) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

const normalize = (s) =>
  String(s ?? "").toLowerCase().replace(/[.'’-]/g, " ").replace(/\s+/g, " ").trim();

function setHTML(id, html) {
  const node = $(id);
  if (node) node.innerHTML = html ?? "";
}

function toast(message, kind = "ok") {
  const wrap = $("#toast-wrap");
  const node = document.createElement("div");
  node.className = `toast toast-${kind}`;
  node.textContent = message;
  wrap.appendChild(node);
  setTimeout(() => {
    node.classList.add("fade-out");
    setTimeout(() => node.remove(), 350);
  }, 3800);
}

/* ------------------------------------------------------------------ */
/* State                                                              */
/* ------------------------------------------------------------------ */

const state = {
  players: [],
  lineups: null,
  app: null,
  excluded: new Set(),
  purchased: new Set(),
  session: null,
  tab: "auction",
  auction: {
    query: "",
    selected: null,
    advice: null,
    price: 1,
    finalPrice: 1,
    buyer: "",
    tier: "B",
    note: "",
  },
  playersTab: {
    query: "",
    selected: null,
    role: "Tutti",
    cluster: "Tutti",
    checked: new Set(),
  },
  lineupsTab: { team: "Tutte", focus: "Tutte", query: "" },
  setup: { budget: 500, fair: null, weights: null, method: "blend" },
};

/* ------------------------------------------------------------------ */
/* API / data loading                                                 */
/* ------------------------------------------------------------------ */

async function loadState() {
  const data = await getJSON("/api/state");
  state.app = data;
  state.excluded = new Set(data.excluded || []);
  state.purchased = new Set(data.purchased || []);
  state.session = data.session;
  if (data.session) {
    state.setup.budget = data.session.budget;
    state.setup.fair = structuredClone(data.session.fair);
  } else {
    state.setup.budget = data.default_budget;
    state.setup.fair = structuredClone(data.fair_default);
  }
  state.setup.weights = structuredClone(data.weights);
  state.setup.method = data.method;
  return data;
}

async function loadPlayers() {
  const data = await getJSON("/api/players");
  state.players = data.players;
}

async function loadLineups() {
  const data = await getJSON("/api/lineups");
  state.lineups = data;
}

async function loadAll() {
  try {
    const [stateData] = await Promise.all([loadState(), loadPlayers(), loadLineups()]);
    setConnection(true);
    renderAll(stateData);
  } catch (err) {
    setConnection(false);
    toast(err.message, "err");
  }
}

async function refreshState({ rerender = true } = {}) {
  try {
    const data = await loadState();
    if (rerender) renderAll(data);
  } catch (err) {
    toast(err.message, "err");
  }
}

function setConnection(ok) {
  const node = $("#connection");
  node.className = ok ? "badge badge-ok" : "badge badge-error";
  node.textContent = ok ? "● connesso" : "● offline";
}

/* ------------------------------------------------------------------ */
/* Helpers used across renderers                                      */
/* ------------------------------------------------------------------ */

const ROLE_ORDER = ["P", "D", "C", "A"];

function availablePlayers() {
  const unavailable = new Set([...state.excluded, ...state.purchased]);
  return state.players.filter((p) => !unavailable.has(p.Nome));
}

function roleProgress(own, slots) {
  return ["P", "D", "C", "A"].map((role) => {
    const filled = (own?.by_role || {})[role] || 0;
    const total = slots?.[role] || 0;
    const ratio = total ? filled / total : 0;
    return { role, filled, total, ratio };
  });
}

function fairValue(role, cluster) {
  const table = state.setup.fair?.[role] || [];
  const index = Math.max(0, Math.min(Number(cluster) - 1, table.length - 1));
  return table[index] ?? 1;
}

function searchPlayers(query, pool) {
  const q = normalize(query);
  if (!q || q.length < 2) return [];
  const exact = [];
  const fuzzy = [];
  for (const p of pool) {
    const name = String(p.Nome).toLowerCase();
    const norm = normalize(p.Nome);
    if (name.includes(q) || norm.includes(q)) {
      const tokens = norm.split(" ");
      const isExact = tokens.includes(q) || (q.length >= 3 && tokens.some((t) => t.startsWith(q)));
      (isExact ? exact : fuzzy).push(p);
    }
  }
  const byQuality = (a, b) =>
    (Number(b.FM) || 0) - (Number(a.FM) || 0) ||
    a.Nome.localeCompare(b.Nome);
  exact.sort(byQuality);
  fuzzy.sort(byQuality);
  return [...exact, ...fuzzy].slice(0, 8);
}

function suggestionsHtml(items, activeIndex, action = "pick") {
  if (!items.length) return '<div class="suggestion-item">Nessun match, prova altro nome</div>';
  return items
    .map(
      (p, i) =>
        `<div class="suggestion-item${i === activeIndex ? " is-active" : ""}" data-action="${action}" data-name="${esc(p.Nome)}" data-index="${i}">
          <span><b>${esc(p.Nome)}</b> <span class="s-meta">${esc(p.Squadra)}</span></span>
          <span class="s-meta"><span class="s-tag">${esc(p.Ruolo)}</span> FM ${num(p.FM)}</span>
        </div>`,
    )
    .join("");
}

/* ------------------------------------------------------------------ */
/* Hero / header                                                      */
/* ------------------------------------------------------------------ */

function renderHero(data) {
  const session = data.session;
  const own = data.own;
  $("#budget-chip").textContent = session
    ? `Budget ${session.budget} cr`
    : "Nessuna configurazione";
  const set = (key, text) => {
    const el = document.querySelector(`[data-stat="${key}"]`);
    if (el) el.textContent = text;
  };
  set("players", int(data.summary.players));
  set("budget", session ? `${int(session.budget)} cr` : "—");
  set("remaining", own ? int(own.remaining) : "—");
  set("bought", own ? String(own.purchases.length) : "—");
}

/* ------------------------------------------------------------------ */
/* ASTA LIVE                                                          */
/* ------------------------------------------------------------------ */

function renderAuction() {
  const session = state.session;
  $("#auction-no-session").classList.toggle("hidden", !!session);
  $$("#tab-auction .panel, #tab-auction .grid").forEach((el) => el.classList.toggle("hidden", !session));
  if (!session) return;
  const own = state.app.own;
  const alerts = state.app.alerts || [];
  const market = state.app.market || [];

  $("#auction-session-name").textContent = `configurazione ${esc(session.name || "")} · ${esc(session.created.replace("T", " "))}`;

  setHTML("#own-metrics", `
    <div class="metric-card"><div class="metric-label">La mia squadra</div><div class="metric-value">${esc(own.team)}</div></div>
    <div class="metric-card"><div class="metric-label">Crediti rimasti</div><div class="metric-value">${int(own.remaining)}</div></div>
    <div class="metric-card"><div class="metric-label">Spesi</div><div class="metric-value">${int(own.spent)}</div></div>
    <div class="metric-card"><div class="metric-label">Giocatori presi</div><div class="metric-value">${own.purchases.length}</div></div>
  `);

  const progress = roleProgress(own, session.slots);
  setHTML("#role-progress", progress.map((r) => `
    <div class="role-prog">
      <div class="rp-top"><span>${esc(r.role)}</span><strong>${r.filled}/${r.total}</strong></div>
      <div class="progress-track"><span style="width:${Math.round(r.ratio * 100)}%"></span></div>
    </div>`).join(""));

  const freshness = (state.app.freshness || [])
    .map((f) => `<span class="chip chip-${f.state}">${f.label}: ${esc(f.value)}</span>`)
    .join("");
  setHTML("#freshness", freshness);

  setHTML("#roster-alerts", alerts.length
    ? `<div class="alert alert-warn"><strong>Attenzioni sulla mia rosa</strong><ul>${alerts.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div>`
    : "");

  renderPlan();
  renderWatchlist();
  renderTeamsEditor();
  renderBoard(market);
  renderRoster();

  const availableCount = availablePlayers().length;
  const note = availableCount
    ? `<p class="note">${availableCount} giocatori disponibili · ${state.purchased.size} già registrati nell'asta.</p>`
    : `<div class="alert alert-info">Non ci sono giocatori disponibili: controlla le esclusioni o annulla un acquisto.</div>`;
  setHTML("#auction-available-note", note);

  renderAuctionSearch();
  renderAuctionAdvice();
}

function renderPlan() {
  const session = state.session;
  if (!session) return;
  const choices = state.app.role_plan_choices;
  const labels = Object.keys(choices);
  const current = session.role_priorities || {};
  const closest = (value) =>
    labels.reduce((best, l) =>
      Math.abs(choices[l] - value) < Math.abs(choices[best] - value) ? l : best, labels[0]);

  setHTML("#plan-controls", ["P", "D", "C", "A"].map((role) => `
    <div class="form-row">
      <label>${esc(role)}: strategia</label>
      <select class="select plan-select" data-role="${esc(role)}">
        ${labels.map((l) => `<option value="${choices[l]}" ${closest(current[role] ?? 1) === l ? "selected" : ""}>${esc(l)}</option>`).join("")}
      </select>
    </div>`).join(""));
}

function renderWatchlist() {
  const session = state.session;
  const watchlist = session?.watchlist || [];
  const explanations = state.app.watchlist_explanations || {};
  if (!watchlist.length) {
    setHTML("#watchlist-panel", `<p class="note">Nessun obiettivo salvato: aggiungine uno dalla scheda del giocatore chiamato.</p>`);
    return;
  }
  setHTML("#watchlist-panel", `
    <div class="table-wrap"><table class="data">
      <thead><tr><th>Giocatore</th><th>Squadra</th><th>Ruolo</th><th>Priorità</th><th>Nota</th></tr></thead>
      <tbody>${watchlist.map((w) => `
        <tr>
          <td>${esc(w.name)}</td><td>${esc(w.club)}</td><td>${esc(w.role)}</td>
          <td><span class="s-tag">${esc(w.tier)}</span> ${esc(explanations[w.tier] || "")}</td><td>${esc(w.note)}</td>
        </tr>`).join("")}
      </tbody>
    </table></div>
    <div class="form-row">
      <label>Rimuovi dalla watchlist</label>
      <select class="select" id="watchlist-remove">
        ${watchlist.map((w) => `<option value="${esc(w.name)}">${esc(w.name)}</option>`).join("")}
      </select>
      <button class="btn btn-danger btn-sm" data-action="remove-watch">Rimuovi obiettivo</button>
    </div>`);
}

function renderTeamsEditor() {
  const session = state.session;
  if (!session) return;
  const teams = session.teams;
  setHTML("#teams-editor", teams.map((team, i) => `
    <input class="input team-input" data-index="${i}" value="${esc(team)}" placeholder="Partecipante ${i + 1}">`).join(""));
  setHTML("#my-team-select", teams.map((team) =>
    `<option value="${esc(team)}" ${team === session.my_team ? "selected" : ""}>${esc(team)}</option>`).join(""));
  setHTML("#roster-team-select", teams.map((team) =>
    `<option value="${esc(team)}">${esc(team)}</option>`).join(""));
}

function purchasesByTeam() {
  const map = {};
  for (const p of state.session?.purchases || []) {
    (map[p.team] = map[p.team] || []).push(p);
  }
  return map;
}

function renderBoard(market) {
  const marketText = (market || [])
    .filter((item) => item.count)
    .map((item) => `${item.role}: ${item.delta >= 0 ? "+" : ""}${(item.delta * 100).toFixed(0)}% su cap (${item.count} acquisti)`)
    .join(" · ");
  $("#market-thermometer").textContent = marketText || "";

  const byTeam = purchasesByTeam();
  const rows = state.session.teams.map((team) => {
    const summary = teamSummary(team, byTeam[team] || []);
    return { team, ...summary };
  });
  setHTML("#board-table", `
    <thead><tr>
      <th>Partecipante</th><th class="num">Crediti rimasti</th><th class="num">Spesi</th>
      <th class="num">Rosa</th><th class="num">P</th><th class="num">D</th><th class="num">C</th><th class="num">A</th>
    </tr></thead>
    <tbody>${rows.map((r) => `
      <tr>
        <td>${esc(r.team)}${r.team === state.session.my_team ? ' <span class="s-tag">io</span>' : ""}</td>
        <td class="num">${int(r.remaining)}</td><td class="num">${int(r.spent)}</td>
        <td class="num">${r.count}</td>
        <td class="num">${r.by_role.P}</td><td class="num">${r.by_role.D}</td>
        <td class="num">${r.by_role.C}</td><td class="num">${r.by_role.A}</td>
      </tr>`).join("")}
    </tbody>`);
}

function teamSummary(team, purchases) {
  const budget = state.session.budget;
  const spent = purchases.reduce((s, p) => s + Number(p.price), 0);
  const byRole = { P: 0, D: 0, C: 0, A: 0 };
  purchases.forEach((p) => { if (byRole[p.role] !== undefined) byRole[p.role] += 1; });
  return { remaining: budget - spent, spent, count: purchases.length, by_role: byRole };
}

function renderRoster() {
  const team = $("#roster-team-select")?.value || state.session.teams[0];
  const purchases = purchasesByTeam()[team] || [];
  if (!purchases.length) {
    setHTML("#roster-view", `<p class="note">${esc(team)} non ha ancora acquisti registrati.</p>`);
    return;
  }
  const rows = [...purchases].reverse().map((p) => `
    <tr>
      <td>${esc(p.name)}</td><td>${esc(p.club)}</td><td>${esc(p.role)}</td>
      <td class="num">${int(p.price)}</td><td class="num">${int(p.cap)}</td>
      <td class="num">${int(p.cluster)}</td><td>${esc((p.created || "").replace("T", " "))}</td>
    </tr>`).join("");
  const undoOptions = [...purchases].reverse()
    .map((p) => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join("");
  setHTML("#roster-view", `
    <div class="table-wrap"><table class="data">
      <thead><tr><th>Giocatore</th><th>Squadra</th><th>Ruolo</th><th class="num">Prezzo</th><th class="num">Cap</th><th class="num">Cluster</th><th>Registrato</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <div class="form-row">
      <label>Annulla acquisto</label>
      <select class="select" id="undo-purchase">${undoOptions}</select>
      <button class="btn btn-danger btn-sm" data-action="undo-purchase">↩️ Annulla acquisto selezionato</button>
    </div>`);
}

function renderAuctionSearch() {
  const query = state.auction.query;
  const available = availablePlayers();
  const matches = searchPlayers(query, available);
  const box = $("#auction-suggestions");
  box.hidden = !(query.length >= 2);
  if (query.length >= 2) {
    box.innerHTML = suggestionsHtml(matches, -1, "pick-auction");
  }
}

function renderAuctionAdvice() {
  const advice = state.auction.advice;
  const session = state.session;
  if (!advice || !session) {
    setHTML("#auction-advice", `<p class="note">Scrivi almeno due lettere e seleziona il giocatore chiamato per vedere il consiglio.</p>`);
    return;
  }
  setHTML("#auction-advice", `
    <div class="form-row price-row">
      <label for="auction-current-price">Prezzo attuale</label>
      <input class="input" id="auction-current-price" type="number" min="1" max="${int(session.budget)}" step="1" value="${int(state.auction.price)}">
      <span class="note">modifica il prezzo: il consiglio si aggiorna subito</span>
    </div>
    <div id="advice-core">${adviceCoreHtml()}</div>
  `);
}

function adviceCoreHtml() {
  const advice = state.auction.advice;
  if (!advice) return "";
  const player = advice.player;
  const a = advice.advice;
  const session = state.session;
  const explanations = state.app.watchlist_explanations || {};
  const verdictClass =
    a.verdict === "PUNTA" ? "verdict-punta"
    : a.verdict === "SOLO SE È UNA PRIORITÀ" ? "verdict-warn"
    : "verdict-leave";
  const watchTier = a.watch_tier ? `<div class="alert alert-info">Watchlist ${esc(a.watch_tier)}: ${esc(explanations[a.watch_tier] || "")}</div>` : "";

  return `
    <div class="bid-card">
      <div>
        <div class="bid-label">Massimo da offrire (STOP)</div>
        <div class="bid-value">${int(a.fixed_cap)} crediti</div>
        <div class="bid-meta">Cluster ${int(player.Cluster)} · fair value del ruolo ${esc(player.Ruolo)}</div>
      </div>
      <div class="bid-stop">STOP oltre ${int(a.fixed_cap)}</div>
    </div>
    <div class="grid cols-3">
      <div class="metric-card"><div class="metric-label">Punta fino a</div><div class="metric-value">${int(a.recommended)} cr</div></div>
      <div class="metric-card"><div class="metric-label">Massimo personale</div><div class="metric-value">${int(a.personal_max)} cr</div></div>
      <div class="metric-card"><div class="metric-label">Crediti rimasti</div><div class="metric-value">${int(state.app.own.remaining)} cr</div><div class="metric-sub">di cui ${int(a.reserve)} riservati agli altri slot</div></div>
    </div>
    <div class="alert ${verdictClass}">
      ${esc(a.verdict)} — prezzo consigliato fino a ${int(a.recommended)} crediti.
    </div>
    <p class="note">
      Prezzo attuale ${int(state.auction.price)} cr · ${int(a.role_left)} slot ${esc(player.Ruolo)} da riempire ·
      riserva ${int(a.reserve)} crediti per gli altri slot · ${int(a.alternatives)} alternative comparabili disponibili.
    </p>
    <p class="note"><strong>Perché:</strong> qualità nel cluster ${pct(a.quality)} · necessità reparto ${pct(a.need)} · strategia ${esc(advice.plan_label)}.</p>
    ${watchTier}

    <div class="panel-head"><h4 style="margin:10px 0 4px">Confronto diretto — il chiamato contro il tuo piano B</h4></div>
    <div class="table-wrap"><table class="data">
      <thead><tr><th>Scelta</th><th>Giocatore</th><th>Squadra</th><th class="num">Cluster</th><th>Valore stagione</th><th>Titolare</th><th class="num">Punta fino a</th><th class="num">STOP</th></tr></thead>
      <tbody>${advice.comparison.map((r) => `
        <tr${r.scelta === "Chiamato ora" ? ' class="selected"' : ""}>
          <td>${esc(r.scelta)}</td><td><b>${esc(r.nome)}</b></td><td>${esc(r.squadra)}</td>
          <td class="num">${int(r.cluster)}</td><td>${pct(r.season_value)}</td><td>${esc(r.starter)}</td>
          <td class="num">${int(r.recommended)}</td><td class="num">${int(r.fixed_cap)}</td>
        </tr>`).join("")}
      </tbody>
    </table></div>
    ${advice.has_alternatives ? "" : `<p class="note">Non ci sono alternative comparabili ancora disponibili in questo ruolo.</p>`}

    <div class="form-row">
      <label>Tipo obiettivo</label>
      <select class="select" id="watch-tier">
        ${Object.keys(explanations).map((t) => `<option value="${t}" ${t === state.auction.tier ? "selected" : ""}>${t} — ${esc(explanations[t])}</option>`).join("")}
      </select>
      <input class="input" id="watch-note" type="text" placeholder="es. titolare" value="${esc(state.auction.note)}">
      <button class="btn btn-primary" data-action="add-watch">⭐ Salva come obiettivo</button>
    </div>

    <div class="panel-head"><h4 style="margin:10px 0 4px">Registra esito della chiamata</h4></div>
    <div class="form-row">
      <label>Aggiudicato a</label>
      <select class="select" id="auction-buyer">
        ${session.teams.map((t) => `<option value="${esc(t)}" ${t === (state.auction.buyer || session.my_team) ? "selected" : ""}>${esc(t)}</option>`).join("")}
      </select>
      <label>Prezzo finale</label>
      <input class="input" id="auction-final-price" type="number" min="1" max="${int(session.budget)}" step="1" value="${state.auction.finalPrice}">
      <button class="btn btn-primary" data-action="record-purchase">✅ Registra acquisto</button>
    </div>`;
}

/* ------------------------------------------------------------------ */
/* GIOCATORI                                                          */
/* ------------------------------------------------------------------ */

function renderPlayers() {
  const hidden = state.excluded.size + state.purchased.size;
  $("#players-hidden-note").textContent = hidden
    ? `${hidden} giocatori non mostrati (già registrati o rimossi)`
    : "";

  const available = availablePlayers();
  const roleFilter = state.playersTab.role;
  const clusterFilter = state.playersTab.cluster;
  let view = roleFilter === "Tutti" ? available : available.filter((p) => p.Ruolo === roleFilter);
  const clusters = [...new Set(view.map((p) => Number(p.Cluster)).filter((c) => c > 0))].sort((a, b) => a - b);
  if (clusterFilter !== "Tutti") view = view.filter((p) => Number(p.Cluster) === Number(clusterFilter));
  view.sort((a, b) => {
    const ia = ROLE_ORDER.indexOf(a.Ruolo);
    const ib = ROLE_ORDER.indexOf(b.Ruolo);
    if (ia !== ib) return ia - ib;
    return Number(a.Rank) - Number(b.Rank);
  });

  $("#players-role").innerHTML =
    ['<option value="Tutti">Tutti</option>',
      ...["P", "D", "C", "A"].map((r) => `<option value="${r}" ${r === roleFilter ? "selected" : ""}>${r}</option>`)].join("");
  $("#players-cluster").innerHTML =
    ['<option value="Tutti">Tutti</option>',
      ...clusters.map((c) => `<option value="${c}" ${String(c) === String(clusterFilter) ? "selected" : ""}>${c}</option>`)].join("");
  $("#players-count-note").textContent = `${view.length} giocatori visibili · budget ${int(state.setup.budget)}`;

  const maxScore = view.length ? Math.max(...view.map((p) => Number(p.Score) || 0)) : null;
  setHTML("#players-table-wrap", `
    <table class="data">
    <thead><tr>
      <th></th><th>Nome</th><th>Squadra</th><th>Ruolo</th><th class="num">FM</th><th class="num">QA</th>
      <th class="num">Cluster</th><th class="num">Score</th><th>Valore stag.</th><th>Confidenza</th><th class="num">Max da offrire</th>
    </tr></thead>
    <tbody>${view.map((p) => `
      <tr>
        <td><input type="checkbox" class="exclude-check" data-name="${esc(p.Nome)}" ${state.playersTab.checked.has(p.Nome) ? "checked" : ""}></td>
        <td><button class="link-inline" data-action="pick-player" data-name="${esc(p.Nome)}">${esc(p.Nome)}</button></td>
        <td>${esc(p.Squadra)}</td><td>${esc(p.Ruolo)}</td>
        <td class="num">${num(p.FM)}</td><td class="num">${int(p.QA)}</td>
        <td class="num">${int(p.Cluster)}</td><td class="num">${num(p.Score, 3)}</td>
        <td>${pct(p.SeasonValue)}</td><td>${pct(p.DataConfidence)}</td>
        <td class="num"><span class="chip chip-green">${int(fairValue(p.Ruolo, p.Cluster))}</span></td>
      </tr>`).join("")}
    </tbody>
    </table>`);

  if (!view.length) setHTML("#players-table-wrap", `<p class="note">Nessun giocatore corrisponde ai filtri.</p>`);

  setHTML("#restore-panel", `
    <details class="accordion panel" style="box-shadow:none;margin:12px 0 0">
      <summary>♻️ Ripristina giocatori rimossi (${state.excluded.size})</summary>
      <div class="accordion-body">
        ${state.excluded.size
          ? `<div class="chips-row">${[...state.excluded].sort().map((n) => `<button class="chip" data-action="restore-player" data-name="${esc(n)}">${esc(n)} ✕</button>`).join("")}</div>`
          : `<p class="note">Nessun giocatore rimosso.</p>`}
      </div>
    </details>`);

  setHTML("#fair-cluster-note", `cluster fissi da 10 giocatori (10 partecipanti, uno a testa per cluster)`);
  setHTML("#fair-tables", ["P", "D", "C", "A"].map((r) => `
    <div class="fair-col">
      <h4>${esc(r)}</h4>
      ${(state.setup.fair?.[r] || []).map((value, i) => `
        <div class="fair-row"><label>Cluster ${i + 1}</label><span>${int(value)}</span></div>`).join("")}
    </div>`).join(""));
  void maxScore;
}

function renderPlayerCard() {
  const selected = state.playersTab.selected;
  if (!selected) {
    setHTML("#player-card", "");
    return;
  }
  const p = selected;
  const fmDisp = p.FM != null ? num(p.FM) : (p.FMEst != null ? `~${num(p.FMEst)} (stima)` : "—");
  const cap = fairValue(p.Ruolo, p.Cluster);
  const roleCount = state.players.filter((x) => x.Ruolo === p.Ruolo).length;
  const components = [
    ["Fantamedia (FCP)", p.C_FM], ["FVM (Gazzetta)", p.C_FVM], ["Algoritmo FCP", p.C_ALG],
    ["Titolare", p.C_Starter], ["Set-pieces", p.C_SetPieces], ["Attributi", p.C_Tags],
    ["Robustezza", p.C_Injury], ["Affidabilità impiego", p.C_Availability],
    ["Modello FM attesa", p.C_Model], ["xG + xA per 90", p.C_ExpectedOutput],
    ["Gol subiti/90 (inv.)", p.C_GolSubiti],
  ].filter(([label, value]) => value != null && (label === "Gol subiti/90 (inv.)" ? value !== 0 : true));
  const maxC = Math.max(1, ...components.map(([, v]) => Number(v) || 0));

  setHTML("#player-card", `
    <div class="bid-card">
      <div>
        <div class="bid-label">${esc(p.Nome)} · ${esc(p.Squadra)}</div>
        <div class="bid-value">${int(cap)} crediti</div>
        <div class="bid-meta">Cluster ${int(p.Cluster)} dei ${esc(p.Ruolo)} · STOP oltre ${int(cap)}</div>
      </div>
      <div class="bid-stop">${esc(p.Ruolo)}</div>
    </div>
    <div class="kv-list">
      <div class="kv"><div class="kv-label">FantaMedia</div><div class="kv-value">${fmDisp}</div></div>
      <div class="kv"><div class="kv-label">Quotazione QA</div><div class="kv-value">${int(p.QA)}</div></div>
      <div class="kv"><div class="kv-label">FM attesa (modello)</div><div class="kv-value">${p.PredFM != null ? num(p.PredFM) : "—"}</div></div>
      <div class="kv"><div class="kv-label">Confidenza dati</div><div class="kv-value">${pct(p.DataConfidence)}</div></div>
      <div class="kv"><div class="kv-label">Affidabilità impiego</div><div class="kv-value">${pct(p.Availability)}</div></div>
      <div class="kv"><div class="kv-label">Valore stagione</div><div class="kv-value">${pct(p.SeasonValue)}</div></div>
      <div class="kv"><div class="kv-label">Upside</div><div class="kv-value">${pct(p.Upside)}</div></div>
      <div class="kv"><div class="kv-label">Rank</div><div class="kv-value">${int(p.Rank)}/${roleCount}</div></div>
    </div>
    ${p.Minuti != null || p.xGI90 != null || p.GolSubiti90 != null ? `
    <div class="kv-list">
      <div class="kv"><div class="kv-label">Minuti storici</div><div class="kv-value">${int(p.Minuti)}</div></div>
      <div class="kv"><div class="kv-label">Titolare storico</div><div class="kv-value">${int(p.Titolarita)}</div></div>
      <div class="kv"><div class="kv-label">xG/90</div><div class="kv-value">${p.xG90 != null ? num(p.xG90) : "—"}</div></div>
      <div class="kv"><div class="kv-label">xA/90</div><div class="kv-value">${p.xA90 != null ? num(p.xA90) : "—"}</div></div>
      <div class="kv"><div class="kv-label">xG+xA/90</div><div class="kv-value">${p.xGI90 != null ? num(p.xGI90) : "—"}</div></div>
      ${p.Ruolo === "P" && (p.GolSubiti90 != null || p.RigoriParati != null) ? `
      <div class="kv"><div class="kv-label">Gol subiti</div><div class="kv-value">${int(p.GolSubiti)}</div></div>
      <div class="kv"><div class="kv-label">Gol subiti/partita</div><div class="kv-value">${p.GolSubiti90 != null ? num(p.GolSubiti90) : "—"}</div></div>
      <div class="kv"><div class="kv-label">Rigori parati</div><div class="kv-value">${p.RigoriParati != null ? int(p.RigoriParati) : "—"}</div></div>
      <div class="kv" style="grid-column:1/-1"><div class="kv-label">Ultime stagioni Serie A (fantacalcio.it)</div><div class="kv-value" style="font-weight:500;font-size:11px;color:var(--muted)">gol subiti totali nelle stagioni scaricate; assente se non in Serie A</div></div>` : ""}
      ${p.GareSaltate != null ? `<div class="kv"><div class="kv-label">Gare saltate</div><div class="kv-value">${int(p.GareSaltate)}</div></div>` : ""}
    </div>` : ""}
    <p class="note">${esc(p.Nome)} è nel cluster ${int(p.Cluster)} dei ${esc(p.Ruolo)} (ordinati per punteggio). Fair value del cluster: ${int(cap)}. Se il prezzo sale sopra ${int(cap)}, esci dall'asta.</p>
    <details class="accordion">
      <summary>🧮 Breakdown punteggio (totale ${num(p.Score, 3)})</summary>
      <div class="accordion-body">
        <div class="breakdown-bar">
          ${components.map(([label, value]) => {
            const share = Math.max(2, (Number(value) / maxC) * 100);
            return `
              <div class="bb-row">
                <span class="bb-label" title="${esc(label)}">${esc(label)}</span>
                <span class="bb-track"><span class="bb-fill" style="width:${share}%"></span></span>
                <span class="bb-val">${num(value, 3)}<small>/${String(Number(maxC))}</small></span>
              </div>`;
          }).join("")}
        </div>
      </div>
    </details>
    <details class="accordion">
      <summary>📋 Tabella fair value per il ruolo</summary>
      <div class="accordion-body">
        <div class="table-wrap"><table class="data">
          <thead><tr><th class="num">Cluster</th><th class="num">Fair value</th></tr></thead>
          <tbody>${(state.setup.fair?.[p.Ruolo] || []).map((v, i) => `<tr><td class="num">${i + 1}</td><td class="num">${int(v)}</td></tr>`).join("")}</tbody>
        </table></div>
      </div>
    </details>`);
}

function renderPlayersSearch() {
  const query = state.playersTab.query;
  const matches = searchPlayers(query, availablePlayers());
  const box = $("#players-suggestions");
  box.hidden = !(query.length >= 2);
  if (query.length >= 2) box.innerHTML = suggestionsHtml(matches, -1, "pick-player");
}

/* ------------------------------------------------------------------ */
/* FORMAZIONI                                                         */
/* ------------------------------------------------------------------ */

function renderLineups() {
  const data = state.lineups;
  if (!data || !data.lineups.length) {
    setHTML("#lineups-list", `<div class="alert alert-info">Nessuna formazione — usa 'Aggiorna tutto per l'asta' nella tab Setup.</div>`);
    return;
  }
  $("#lineups-updated").textContent = data.updated_at ? `Ultimo aggiornamento: ${esc(data.updated_at)}` : "";

  setHTML("#lineups-legend", `
    <span><span class="dot" style="background:#c62828"></span>⚽ Rigorista</span>
    <span><span class="dot" style="background:#1565c0"></span>🚩 Angoli</span>
    <span><span class="dot" style="background:#b8860b"></span>🎯 Piazzati</span>
    <span>il numero indica la posizione nella gerarchia</span>
    <span>⭐ = nome da monitorare per bonus</span>
    <span>* = fantamedia stimata</span>
  `);

  const teams = data.teams;
  const attentionTeams = new Set(data.attention_teams);
  const bonusTeams = new Set(data.bonus_teams);
  setHTML("#lineups-overview", `
    <div class="stat-card" style="background:var(--panel-2);border-color:var(--border)"><span class="stat-label">Squadre</span><strong class="stat-value">${teams.length}</strong></div>
    <div class="stat-card" style="background:var(--panel-2);border-color:var(--border)"><span class="stat-label">Titolari attesi</span><strong class="stat-value">${data.lineups.length}</strong></div>
    <div class="stat-card" style="background:var(--panel-2);border-color:var(--border)"><span class="stat-label">Squadre con bonus</span><strong class="stat-value">${bonusTeams.size}</strong></div>
    <div class="stat-card" style="background:var(--panel-2);border-color:var(--border)"><span class="stat-label">Da monitorare</span><strong class="stat-value">${attentionTeams.size}</strong></div>
  `);

  $("#lineups-team").innerHTML =
    ['<option value="Tutte">Tutte</option>',
      ...teams.map((t) => `<option value="${esc(t)}" ${t === state.lineupsTab.team ? "selected" : ""}>${esc(t)}</option>`)].join("");
  $("#lineups-focus").innerHTML =
    [["Tutte", "Tutte"], ["bonus", "Solo bonus"], ["monitor", "Solo da monitorare"]]
      .map(([value, label]) => `<option value="${value}" ${value === state.lineupsTab.focus ? "selected" : ""}>${esc(label)}</option>`).join("");
  $("#lineups-query").value = state.lineupsTab.query;

  const byTeam = new Map();
  for (const row of data.lineups) {
    if (!byTeam.has(row.Squadra)) byTeam.set(row.Squadra, []);
    byTeam.get(row.Squadra).push(row);
  }

  let shown = teams;
  if (state.lineupsTab.team !== "Tutte") shown = [state.lineupsTab.team];
  if (state.lineupsTab.focus === "bonus") shown = shown.filter((t) => bonusTeams.has(t));
  if (state.lineupsTab.focus === "monitor") shown = shown.filter((t) => attentionTeams.has(t));
  const q = normalize(state.lineupsTab.query);
  if (q) shown = shown.filter((t) => (byTeam.get(t) || []).some((r) => normalize(r.Nome || r.NomeLineup).includes(q)));

  $("#lineups-count-note").textContent = `${shown.length} squadre visibili · aggiorna i dati dalla tab Setup il giorno dell'asta`;
  if (!shown.length) {
    setHTML("#lineups-list", `<div class="alert alert-info">Nessuna squadra corrisponde ai filtri selezionati.</div>`);
    return;
  }

  setHTML("#lineups-list", `<div class="team-grid">${shown.map((team) => renderTeamCard(team, byTeam.get(team) || [])).join("")}</div>`);
}

function renderTeamCard(team, rows) {
  const modulo = rows[0]?.Modulo || "?";
  const sections = [];
  for (const role of ["P", "D", "C", "A", ""]) {
    const roleRows = rows.filter((r) => (r.Ruolo || "") === role);
    if (!roleRows.length) continue;
    const title = role || "Ruolo da verificare";
    const items = roleRows.map(playerRowHtml).join("");
    sections.push(`<div class="role-section-title">${esc(title)}</div>${items}`);
  }

  const noteLabels = [
    ["Ballottaggi", "🔄 Ballottaggi"], ["Squalificati", "⛔ Squalificati"],
    ["Infortunati", "🩹 Infortunati"], ["InDubbio", "❓ In dubbio"], ["Diffidati", "🟨 Diffidati"],
  ];
  const notes = noteLabels
    .map(([field, label]) => {
      const value = rows[0]?.[field];
      return value ? `<li><strong>${label}:</strong> ${esc(value)}</li>` : "";
    })
    .filter(Boolean);
  const notesHtml = notes.length ? `<div class="team-notes alert alert-warn"><ul>${notes.join("")}</ul></div>` : "";

  return `
    <div class="team-card">
      <div class="team-card-head"><span>${esc(team)}</span><span>${esc(modulo)} · ${rows.length} titolari attesi</span></div>
      <div class="team-card-body">
        ${sections.join("")}
        ${notesHtml}
      </div>
    </div>`;
}

function playerRowHtml(row) {
  const flagged = !!(row.Rigorista || row.Punizioni || row.Angoli);
  let badges = "";
  if (row.Rigorista) badges += `<span class="setpiece-badge sp-rig">⚽ ${int(row.RigoristaOrdine)}° rigore</span> `;
  if (row.Angoli) badges += `<span class="setpiece-badge sp-ang">🚩 ${int(row.AngoliOrdine)}° angolo</span> `;
  if (row.Punizioni) badges += `<span class="setpiece-badge sp-pia">🎯 ${int(row.PunizioniOrdine)}° punizione</span> `;
  if (row.Piazzati) badges += `<span class="setpiece-badge sp-pia">🎯 ${int(row.PiazzatiOrdine)}° piazzati</span> `;
  let fm = row.FM != null ? num(row.FM) : "—";
  if (row.FM != null && row.FMImputed) fm += " *";
  const cluster = row.Cluster != null && row.Cluster !== ""
    ? `<span class="cluster-badge">C${int(row.Cluster)}</span>` : "";
  let bench = "";
  if (row.Panchina) {
    const bfm = row.PanchinaFM != null ? `FM ${num(row.PanchinaFM)}` : "FM -";
    const bcl = row.PanchinaCluster != null && row.PanchinaCluster !== ""
      ? ` · C${int(row.PanchinaCluster)}` : "";
    const bruolo = row.PanchinaRuolo ? `${esc(row.PanchinaRuolo)} · ` : "";
    bench = `<div class="pr-bench">⬇ possibile panchinaro: <b>${esc(row.Panchina)}</b> <span class="pr-meta">${bruolo}${bfm}${bcl}</span></div>`;
  }
  return `<div class="player-row ${flagged ? "flagged" : ""}">${cluster}<b>${esc(row.Nome || row.NomeLineup)}</b> <span class="pr-meta">${esc(row.Ruolo || "?")} · FM ${fm}</span> ${badges}${bench}</div>`;
}

/* ------------------------------------------------------------------ */
/* SETUP                                                              */
/* ------------------------------------------------------------------ */

function renderSetup() {
  const data = state.app;
  $("#setup-advanced-note").textContent = data.advanced_present
    ? "Le statistiche avanzate CSV presenti verranno mantenute e non sovrascritte dall'aggiornamento."
    : "Statistiche avanzate non presenti: verranno lasciate vuote dall'aggiornamento.";

  setHTML("#setup-summary", `
    <div class="metric-card"><div class="metric-label">Giocatori</div><div class="metric-value">${int(data.summary.players)}</div></div>
    <div class="metric-card"><div class="metric-label">Con quotazione</div><div class="metric-value">${int(data.summary.with_quote)}</div></div>
    <div class="metric-card"><div class="metric-label">Con fantamedia</div><div class="metric-value">${int(data.summary.with_fm)}</div></div>
    <div class="metric-card"><div class="metric-label">Titolari probabili</div><div class="metric-value">${int(data.summary.starters)}</div></div>
    <div class="metric-card"><div class="metric-label">Con minuti storici</div><div class="metric-value">${int(data.summary.with_minutes)}</div></div>
  `);

  setHTML("#setup-preview", `
    <thead><tr><th>Nome</th><th>Squadra</th><th>Ruolo</th><th class="num">FM</th><th class="num">QA</th><th class="num">QI</th><th class="num">Cluster</th><th>Attributi</th><th>ResInf</th></tr></thead>
    <tbody>${state.players.slice(0, 50).map((p) => `
      <tr><td>${esc(p.Nome)}</td><td>${esc(p.Squadra)}</td><td>${esc(p.Ruolo)}</td>
      <td class="num">${num(p.FM)}</td><td class="num">${int(p.QA)}</td><td class="num">${int(p.QI)}</td>
      <td class="num">${int(p.Cluster)}</td><td>${esc(p.Attributi)}</td><td>${esc(p.ResInf)}</td></tr>`).join("")}
    </tbody>`);

  $("#setup-budget").value = state.setup.budget;

  const sessionOptions = ['<option value="">— nessuna —</option>',
    ...(data.sessions || []).map((s) => `<option value="${esc(s.name)}" ${s.name === data.active_name ? "selected" : ""}>${esc(s.label)}</option>`)].join("");
  $("#setup-load-session").innerHTML = sessionOptions;

  renderFairEditor();
  renderWeightsEditor();

  setHTML("#rho-chips", ["P", "D", "C", "A"].map((role) => {
    const v = data.rho?.[role];
    return `<span class="chip ${v == null ? "chip-muted" : Number(v) >= 0.3 ? "chip-green" : "chip-amber"}">${esc(role)}: ${v == null ? "n/d" : num(v, 2)}</span>`;
  }).join(""));
}

function renderFairEditor() {
  $("#setup-budget").value = state.setup.budget;
  setHTML("#fair-editor", ["P", "D", "C", "A"].map((role) => `
    <div class="fair-col">
      <h4>${esc(role)}</h4>
      ${(state.setup.fair?.[role] || []).map((value, i) => `
        <div class="fair-row"><label>Slot ${i + 1}</label><input class="input fair-input" data-role="${esc(role)}" data-index="${i}" type="number" min="1" value="${int(value)}"></div>`).join("")}
    </div>`).join(""));
}

function renderWeightsEditor() {
  const labels = state.app.weight_labels || {};
  $("#setup-method").innerHTML = Object.entries(state.app.method_labels || {})
    .map(([value, label]) => `<option value="${value}" ${value === state.setup.method ? "selected" : ""}>${esc(label)}</option>`)
    .join("");
  const weightKeys = Object.keys(labels);
  setHTML("#weights-editor", weightKeys.map((key) => {
    const value = Number(state.setup.weights?.[key] ?? 0);
    return `
      <div class="slider-row">
        <div class="slider-top"><span>${esc(labels[key])}</span><span>${num(value, 2)}</span></div>
        <input type="range" class="weight-slider" data-key="${esc(key)}" min="0" max="2" step="0.05" value="${value}">
      </div>`;
  }).join(""));
}

function renderScrapeStatus(status) {
  if (!status || !status.running) {
    $("#scrape-progress").classList.add("hidden");
    setHTML("#scrape-status", "");
    return;
  }
  $("#scrape-progress").classList.remove("hidden");
  const pct = Math.round(status.progress * 100);
  $("#scrape-progress").querySelector("span").style.width = `${pct}%`;
  setHTML("#scrape-status", `<span class="chip chip-green">${esc(status.label)}</span>`);
}

/* ------------------------------------------------------------------ */
/* Global render                                                      */
/* ------------------------------------------------------------------ */

function renderAll(data) {
  renderHero(data);
  switch (state.tab) {
    case "auction": renderAuction(); break;
    case "players": renderPlayers(); renderPlayerCard(); break;
    case "lineups": renderLineups(); break;
    case "setup": renderSetup(); break;
  }
}

function switchTab(tab) {
  state.tab = tab;
  $$("#tab-nav .tab").forEach((b) => b.classList.toggle("is-active", b.dataset.tab === tab));
  $$(".tab-panel").forEach((p) => p.classList.toggle("is-active", p.dataset.panel === tab));
  if (state.app) renderAll(state.app);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ------------------------------------------------------------------ */
/* Async actions                                                      */
/* ------------------------------------------------------------------ */

async function runScrape() {
  try {
    const data = await postJSON("/api/scrape/start", {});
    toast(data.started ? "Aggiornamento avviato in background." : "Aggiornamento già in corso.", data.started ? "ok" : "warn");
    pollScrape();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function pollScrape() {
  try {
    const status = await getJSON("/api/scrape/status");
    renderScrapeStatus(status);
    if (status.running) {
      setTimeout(pollScrape, 1500);
    } else if (status.done) {
      $("#scrape-progress").classList.add("hidden");
      setHTML("#scrape-status", "");
      if (status.error) {
        toast(status.error, "err");
      } else {
        toast("Aggiornamento completato. Ranking e cluster ricalcolati.", "ok");
        await loadPlayers();
        await refreshState();
      }
    }
  } catch (err) {
    toast(err.message, "err");
  }
}

async function saveConfig() {
  const budget = Math.max(10, parseInt($("#setup-budget").value, 10) || 500);
  const fair = {};
  $$(".fair-input").forEach((input) => {
    const role = input.dataset.role;
    const index = Number(input.dataset.index);
    fair[role] = fair[role] || [];
    fair[role][index] = Math.max(1, parseInt(input.value, 10) || 1);
  });
  try {
    await postJSON("/api/session/save", { budget, fair });
    toast("Configurazione salvata.", "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function loadConfig(name) {
  if (!name) return;
  try {
    await postJSON("/api/session/load", { name });
    toast("Configurazione caricata.", "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function savePlan() {
  const priorities = {};
  $$(".plan-select").forEach((select) => {
    priorities[select.dataset.role] = Number(select.value);
  });
  try {
    await postJSON("/api/session/plan", { priorities });
    toast("Piano personale salvato.", "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function saveTeams() {
  const teams = $$(".team-input").map((input) => input.value.trim());
  const myTeam = $("#my-team-select").value;
  try {
    await postJSON("/api/session/teams", { teams, my_team: myTeam });
    toast("Partecipanti salvati.", "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function saveWeights() {
  const weights = {};
  $$(".weight-slider").forEach((slider) => {
    weights[slider.dataset.key] = Number(slider.value);
  });
  const method = $("#setup-method").value;
  try {
    const data = await postJSON("/api/weights", { weights, method });
    state.setup.weights = data.weights;
    state.setup.method = method;
    toast("Pesi salvati — classifica ricalcolata.", "ok");
    await loadPlayers();
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function importAdvanced() {
  const file = $("#advanced-file").files[0];
  if (!file) {
    toast("Seleziona un file CSV.", "warn");
    return;
  }
  try {
    const data = await postForm("/api/advanced", file);
    toast(`Importati ${data.imported} giocatori (${data.with_minutes} con minuti). Ranking aggiornato.`, "ok");
    await loadPlayers();
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function excludeSelected() {
  const names = [...state.playersTab.checked];
  if (!names.length) {
    toast("Nessun giocatore selezionato.", "warn");
    return;
  }
  const merged = new Set([...state.excluded, ...names]);
  try {
    const data = await postJSON("/api/excluded", { names: [...merged] });
    state.excluded = new Set(data.excluded);
    state.playersTab.checked.clear();
    toast(`Rimossi ${names.length} giocatori.`, "ok");
    renderPlayers();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function restorePlayer(name) {
  const next = new Set(state.excluded);
  next.delete(name);
  try {
    const data = await postJSON("/api/excluded", { names: [...next] });
    state.excluded = new Set(data.excluded);
    toast(`${name} ripristinato.`, "ok");
    renderPlayers();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function addWatch() {
  if (!state.auction.advice) return;
  const name = state.auction.advice.player.Nome;
  const tier = $("#watch-tier").value;
  const note = $("#watch-note").value;
  try {
    await postJSON("/api/session/watch_add", { name, tier, note });
    state.auction.tier = tier;
    state.auction.note = note;
    toast(`${name} aggiunto alla watchlist ${tier}.`, "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function removeWatch() {
  const name = $("#watchlist-remove").value;
  if (!name) return;
  try {
    await postJSON("/api/session/watch_remove", { name });
    toast(`Obiettivo ${name} rimosso.`, "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function recordPurchase() {
  const name = state.auction.advice?.player?.Nome;
  if (!name) return;
  const team = $("#auction-buyer").value;
  const price = parseInt($("#auction-final-price").value, 10);
  try {
    const data = await postJSON("/api/session/purchase", { name, team, price });
    let message = `Registrato ${name} a ${team} per ${price} crediti.`;
    if (data.over_cap > 0) message += ` ${data.over_cap} sopra il cap consigliato.`;
    toast(message, "ok");
    state.auction.advice = null;
    state.auction.query = "";
    state.auction.selected = null;
    $("#auction-query").value = "";
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function undoPurchase() {
  const name = $("#undo-purchase").value;
  if (!name) return;
  try {
    await postJSON("/api/session/undo", { name });
    toast(`Acquisto di ${name} annullato.`, "ok");
    await refreshState();
  } catch (err) {
    toast(err.message, "err");
  }
}

async function refreshAuctionAdvice(price) {
  if (!state.auction.advice) return;
  const p = Math.max(1, parseInt(price, 10) || 1);
  state.auction.price = p;
  state.auction.finalPrice = p;
  try {
    const data = await postJSON("/api/advice", {
      name: state.auction.advice.player.Nome,
      current_price: p,
    });
    state.auction.advice = data;
    const core = $("#advice-core");
    if (core) core.innerHTML = adviceCoreHtml();
  } catch (err) {
    toast(err.message, "err");
  }
}

const updateAuctionPrice = debounce((value) => refreshAuctionAdvice(value), 400);

async function pickAuctionPlayer(name) {
  state.auction.selected = name;
  state.auction.query = name;
  state.auction.advice = null;
  const available = availablePlayers();
  const match = available.find((p) => p.Nome === name);
  state.auction.price = 1;
  state.auction.finalPrice = 1;
  state.auction.buyer = state.session?.my_team || "";
  $("#auction-query").value = name;
  $("#auction-suggestions").hidden = true;
  if (match && match.QA != null) state.auction.finalPrice = Math.max(1, Number(match.QA) || 1);
  try {
    const data = await postJSON("/api/advice", { name, current_price: state.auction.price });
    state.auction.advice = data;
    renderAuctionAdvice();
  } catch (err) {
    toast(err.message, "err");
    setHTML("#auction-advice", "");
  }
}

async function pickPlayer(name) {
  state.playersTab.selected = null;
  state.playersTab.query = name;
  const pool = availablePlayers();
  const match = pool.find((p) => p.Nome === name);
  if (match) state.playersTab.selected = match;
  $("#players-query").value = name;
  $("#players-suggestions").hidden = true;
  renderPlayerCard();
}

/* ------------------------------------------------------------------ */
/* Event wiring                                                       */
/* ------------------------------------------------------------------ */

function bindEvents() {
  $("#tab-nav").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) switchTab(tab.dataset.tab);
  });

  $("#theme-toggle").addEventListener("click", () => {
    const light = document.body.classList.toggle("light");
    localStorage.setItem("asta-theme", light ? "light" : "dark");
    $("#theme-toggle").textContent = light ? "🌞 Tema" : "🌙 Tema";
  });

  document.addEventListener("click", async (event) => {
    const actionEl = event.target.closest("[data-action]");
    if (!actionEl) return;
    const action = actionEl.dataset.action;
    const name = actionEl.dataset.name;
    switch (action) {
      case "goto-setup": switchTab("setup"); break;
      case "pick-auction": await pickAuctionPlayer(name); break;
      case "pick-player": await pickPlayer(name); break;
      case "pick": await pickPlayer(name); break;
      case "run-scrape": await runScrape(); break;
      case "import-advanced": await importAdvanced(); break;
      case "save-config": await saveConfig(); break;
      case "save-weights": await saveWeights(); break;
      case "save-plan": await savePlan(); break;
      case "save-teams": await saveTeams(); break;
      case "add-watch": await addWatch(); break;
      case "remove-watch": await removeWatch(); break;
      case "record-purchase": await recordPurchase(); break;
      case "undo-purchase": await undoPurchase(); break;
      case "exclude-selected": await excludeSelected(); break;
      case "restore-player": await restorePlayer(name); break;
    }
  });

  const auctionQuery = $("#auction-query");
  const auctionDebounce = debounce(() => {
    state.auction.query = auctionQuery.value.trim();
    renderAuctionSearch();
  }, 120);
  auctionQuery.addEventListener("input", auctionDebounce);
  auctionQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      const matches = searchPlayers(auctionQuery.value.trim(), availablePlayers());
      if (matches.length) pickAuctionPlayer(matches[0].Nome);
    }
    if (event.key === "Escape") $("#auction-suggestions").hidden = true;
  });

  $("#auction-workspace").addEventListener("input", (event) => {
    if (event.target.id === "auction-current-price") {
      updateAuctionPrice(event.target.value);
    }
  });
  $("#auction-workspace").addEventListener("keydown", (event) => {
    if (event.target.id === "auction-current-price" && event.key === "Enter") {
      refreshAuctionAdvice(event.target.value);
    }
  });

  const playersQuery = $("#players-query");
  const playersDebounce = debounce(() => {
    state.playersTab.query = playersQuery.value.trim();
    renderPlayersSearch();
  }, 120);
  playersQuery.addEventListener("input", playersDebounce);
  playersQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      const matches = searchPlayers(playersQuery.value.trim(), availablePlayers());
      if (matches.length) pickPlayer(matches[0].Nome);
    }
    if (event.key === "Escape") $("#players-suggestions").hidden = true;
  });

  $("#players-role").addEventListener("change", (event) => {
    state.playersTab.role = event.target.value;
    state.playersTab.cluster = "Tutti";
    renderPlayers();
  });
  $("#players-cluster").addEventListener("change", (event) => {
    state.playersTab.cluster = event.target.value;
    renderPlayers();
  });

  $("#players-table-wrap").addEventListener("change", (event) => {
    const box = event.target.closest(".exclude-check");
    if (!box) return;
    if (box.checked) state.playersTab.checked.add(box.dataset.name);
    else state.playersTab.checked.delete(box.dataset.name);
  });

  $("#lineups-team").addEventListener("change", (event) => {
    state.lineupsTab.team = event.target.value;
    renderLineups();
  });
  $("#lineups-focus").addEventListener("change", (event) => {
    state.lineupsTab.focus = event.target.value;
    renderLineups();
  });
  const lineupsQuery = $("#lineups-query");
  const lineupsDebounce = debounce(() => {
    state.lineupsTab.query = lineupsQuery.value.trim();
    renderLineups();
  }, 150);
  lineupsQuery.addEventListener("input", lineupsDebounce);

  $("#setup-budget").addEventListener("change", (event) => {
    const budget = Math.max(10, parseInt(event.target.value, 10) || 500);
    const base = state.app.fair_base;
    state.setup.fair = Object.fromEntries(
      Object.entries(base || {}).map(([role, values]) => [
        role,
        values.map((v) => Math.max(1, Math.round((Number(v) * budget) / 1000))),
      ]),
    );
    state.setup.budget = budget;
    renderFairEditor();
  });

  $("#setup-load-session").addEventListener("change", (event) => {
    loadConfig(event.target.value);
  });

  $("#setup-method").addEventListener("change", (event) => {
    state.setup.method = event.target.value;
  });

  const rosterTeam = $("#roster-team-select");
  rosterTeam.addEventListener("change", () => renderRoster());
}

/* ------------------------------------------------------------------ */
/* Init                                                               */
/* ------------------------------------------------------------------ */

function initTheme() {
  const saved = localStorage.getItem("asta-theme");
  const light = saved === "light";
  document.body.classList.toggle("light", light);
  $("#theme-toggle").textContent = light ? "🌞 Tema" : "🌙 Tema";
}

function initScrollProgress() {
  const bar = $("#scroll-progress-bar");
  const onScroll = () => {
    const total = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = `${total > 0 ? (window.scrollY / total) * 100 : 0}%`;
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

function init() {
  initTheme();
  initScrollProgress();
  bindEvents();
  loadAll();
  setInterval(() => {
    if (document.visibilityState !== "visible") return;
    if (state.tab !== "auction" && state.tab !== "players") return;
    refreshState({ rerender: false }).then(() => renderAll(state.app)).catch(() => {});
  }, 45000);
}

init();
